import torch.optim as optim
import torch
from torchvision.utils import save_image


import os
import Losses
import eval as ev
import numpy as np
from CamPoseNet import CamPoseNet
import utils as utils
from torch.optim.lr_scheduler import ExponentialLR
from torch.nn import functional as F
import torch_bingham


class CamPose():

    def __init__(self, args, device, pretrained=False):
        super(CamPose, self).__init__()
        self.args = args
        self.exp = '{}{}_{}_{}_{}_numcoeff{}/'.format(self.args.save_dir,
                    self.args.dataset, self.args.scene,
                    self.args.base, self.args.loss, str(self.args.num_coeff))

        print(self.exp)
        self.losses = {}
        self.device = device
        self.model = CamPoseNet(args.num_coeff, args.stage, args.base, args.loss, pretrained)

        self.losses['PoseRWTA_KL_CE'] = Losses.rWTALoss_KL_CE(self.args.num_coeff)
        
        self.losses['PoseRWTA_KL_MB'] = Losses.rWTALoss_KL_MB(self.args.num_coeff)

        self.model.to(device)
        
        self.num_steps = self.args.sche_steps
        weight_decay = self.args.adam_delay
        
        if args.stage == 1:
            to_optimize = list(self.model.fc_pose.parameters()) + list(self.model.fc_Z.parameters())
        elif args.stage == 4:
            to_optimize = list(self.model.generator.parameters()) + list(self.model.fc_coeff.parameters())
        else:
            to_optimize = self.model.parameters()

        self.optimizer = optim.Adam(to_optimize, lr=self.args.learning_rate, weight_decay=weight_decay)
        self.scheduler = ExponentialLR(self.optimizer, self.args.sche_eplr)

        if args.restore:
            self.model.load_state_dict(torch.load(self.exp + args.model), strict=False)
            print('Restored model')
        print('Initialized')

    def train(self, train_loader):
        
        log_fd = os.path.join(self.args.save_dir, '{}/{}_{}/'.format(self.args.scene, self.args.loss, self.args.num_coeff))
        if not os.path.exists(log_fd):
            os.makedirs(log_fd)
            
        log_recon = os.path.join(log_fd, 'Recon/')
        if not os.path.exists(log_recon):
            os.makedirs(log_recon)
        
        log_fn = os.path.join(log_fd, '{}_log_stage{}.txt'.format(self.args.scene, self.args.stage))
        fid = open(log_fn, 'w')

        for epoch in range(self.args.num_epochs):

            for batch_idx, (data, target) in enumerate(train_loader):

                self.model.train()
                self.optimizer.zero_grad()

                inputs = data.to(self.device)

                dis_paras, _, x_gen, x_dis = self.model(inputs, True)                
                target = target.to(self.device)

                
                if self.args.loss == 'CE':
                    accR, wloss, accT = self.losses['PoseRWTA_KL_CE'](dis_paras[0], dis_paras[1], dis_paras[2], target[:, 0:4], dis_paras[3], dis_paras[4], target[:, 4:7], dis_paras[5])
                else:
                    accR, wloss, accT = self.losses['PoseRWTA_KL_MB'](dis_paras[0], dis_paras[1], dis_paras[2], target[:, 0:4], dis_paras[3], dis_paras[4], target[:, 4:7])           
                

                mse_loss_gen = F.binary_cross_entropy(x_gen, inputs)
                mse_loss_dis = F.binary_cross_entropy(x_dis, inputs)
                
                # ELBO of VAE
                if self.args.stage == 0:
                    acc = accT + mse_loss_dis
                    print("Epoch: [%2d] [%4d], accT %g, mse_dis %g" % (epoch, batch_idx, accT.item(), mse_loss_dis.item()))
                    fid.write("Epoch: [%2d] [%4d], accT %g, mse_dis %g \n" % (epoch, batch_idx, accT.item(), mse_loss_dis.item()))
                elif self.args.stage == 1:
                    acc = accR + mse_loss_dis
                    print("Epoch: [%2d] [%4d], accR %g, mse_dis %g" % (epoch, batch_idx, accR.item(), mse_loss_dis.item()))
                    fid.write("Epoch: [%2d] [%4d], accR %g, mse_dis %g \n" % (epoch, batch_idx, accR.item(), mse_loss_dis.item()))
                elif self.args.stage == 2:
                    acc = accT + accR + wloss + mse_loss_dis
                    print("Epoch: [%2d] [%4d], accT %g, accR %g, wloss %g, mse_dis %g" % (epoch, batch_idx, accT.item(), accR.item(), wloss.item(), mse_loss_dis.item()))
                    fid.write("Epoch: [%2d] [%4d], accT %g, accR %g, wloss %g, mse_dis %g \n" % (epoch, batch_idx, accT.item(), accR.item(), wloss.item(), mse_loss_dis.item()))
                elif self.args.stage == 3:
                    acc = accT + accR + wloss*50 + mse_loss_gen*50 + mse_loss_dis
                    print("Epoch: [%2d] [%4d], accT %g, accR %g, wloss %g, mse_gen %g, mse_dis %g" % (epoch, batch_idx, accT.item(), accR.item(), wloss.item(), mse_loss_gen.item(), mse_loss_dis.item()))
                    fid.write("Epoch: [%2d] [%4d], accT %g, accR %g, wloss %g, mse_gen %g, mse_dis %g \n" % (epoch, batch_idx, accT.item(), accR.item(), wloss.item(), mse_loss_gen.item(), mse_loss_dis.item())) 
                elif self.args.stage == 4:
                    acc = wloss*50 + mse_loss_gen*50
                    print("Epoch: [%2d] [%4d], wloss %g, mse_gen %g" % (epoch, batch_idx, wloss.item(), mse_loss_gen.item()))
                    fid.write("Epoch: [%2d] [%4d], wloss %g, mse_gen %g \n" % (epoch, batch_idx, wloss.item(), mse_loss_gen.item()))              
                else:
                    raise NotImplementedError()

                acc.backward()
                self.optimizer.step()

                print("Epoch: [%2d] [%4d], training error %g" % (epoch, batch_idx, acc.item()))
                fid.write("Epoch: [%2d] [%4d], training error %g \n" % (epoch, batch_idx, acc.item()))

                self.model.eval()
            
            if self.args.stage >= 3:
                save_image(inputs[0], os.path.join(log_recon, 'img_stage{}_epoch{}_org.png'.format(self.args.stage, epoch)))
                save_image(x_gen[0], os.path.join(log_recon, 'img_stage{}_epoch{}_gen.png'.format(self.args.stage, epoch)))

            outputs = [dis_paras[i].to('cpu').data.numpy() for i in range(len(dis_paras))]
            target = target.to('cpu').data.numpy()


            pred_r,_, pred_t,_, idxs = self.extract_predictions(outputs, target, type=self.args.prediction_type)
            pred_r_oracle,_, pred_t_oracle,_, idxs_gt = self.extract_predictions(outputs, target, type='oracle')
            
            # print(idxs)
            # print(idxs_gt)
                
            print("Match: ", np.sum(idxs == idxs_gt), "/", idxs.shape[0])

            oracle_rotation_error, oracle_translation_error = ev.evaluate_pose_batch(target, pred_r_oracle,
                                                                         pred_t_oracle)

            print(("(Oracle) Median rotation error %f, translation error %f, %f, %f") % (
            np.median(oracle_rotation_error), np.median(oracle_translation_error), np.mean(oracle_translation_error), np.std(oracle_translation_error)))
            fid.write(("(Oracle) Median rotation error %f, translation error %f, %f, %f \n") % (
            np.median(oracle_rotation_error), np.median(oracle_translation_error), np.mean(oracle_translation_error), np.std(oracle_translation_error)))

            rotation_error, translation_error = ev.evaluate_pose_batch(target, pred_r,pred_t)
            print(("Median rotation error %f, translation error %f, %f, %f") % (np.median(rotation_error),
                                                                np.median(translation_error), np.mean(translation_error), np.std(translation_error)))
            fid.write(("Median rotation error %f, translation error %f, %f, %f \n") % (np.median(rotation_error),
                                                                np.median(translation_error), np.mean(translation_error), np.std(translation_error)))
            
            

            if epoch > 0 and epoch % self.num_steps == 0:
                self.scheduler.step()
                
        fid.close()
        
        if not os.path.exists(self.exp):
            os.makedirs(self.exp)
        torch.save(self.model.state_dict(), self.exp + 'model_%d_%d' % (epoch, self.args.stage))
        torch.save(self.optimizer.state_dict(), self.exp + 'optimizer')

    def eval(self, loader):
        results = {}
        results['rotations'] = []
        results['rotations_best'] = []
        results['translations'] = []
        results['translations_best'] = []
        results['labels'] = []
        
        X = []
        Y = []
        Y_ = []
        
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(loader):
                self.model.eval()
                inputs = data.to(self.device)
                dis_paras, _, _, x_dis = self.model(inputs)

                outputs = [dis_paras[i].to('cpu').numpy() for i in range(0, len(dis_paras))]
                target = target.to('cpu').numpy()

                pred_r, lambdas, pred_t, pred_vars, idxs = self.extract_predictions(outputs, target, type=self.args.prediction_type, batch_idx=batch_idx)
                pred_r_oracle, lambdas_oracle, pred_t_oracle, pred_vars_oracle, idxs_gt = self.extract_predictions(outputs, target, type='oracle')
                
                
                X.append(x_dis.to('cpu').numpy())
                Y.append(idxs_gt)   
                Y_.append(idxs)

                # store everything for evaluation
                results['rotations'].extend(pred_r)
                results['rotations_best'].extend(pred_r_oracle)
                results['translations'].extend(pred_t)
                results['translations_best'].extend(pred_t_oracle)
                results['labels'].extend(target)
                
                    
        X = np.vstack(X)
        Y = np.hstack(Y)
        Y_ = np.hstack(Y_)
        
        return results, X, Y, Y_
    
    def eval_recon(self, loader):
        
        log_fd = os.path.join(self.args.save_dir, '{}/{}_{}/'.format(self.args.scene, self.args.loss, self.args.num_coeff))
        if not os.path.exists(log_fd):
            os.makedirs(log_fd)
            
        log_eval_recon = os.path.join(log_fd, 'Recon_eval/')
        if not os.path.exists(log_eval_recon):
            os.makedirs(log_eval_recon)
        
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(loader):
                self.model.eval()
                
                inputs = data.to(self.device)
                target = target.to(self.device)
                
                pose = target[:, 0:4]
                xyz = target[:, 4:7]
                
                x_gen = self.model.recon(pose, xyz)
                
                
                for i in range(inputs.shape[0]):
                    save_image(inputs[i], os.path.join(log_eval_recon, 'img_batch{}_idx{}_org.png'.format(batch_idx, i)))
                    save_image(x_gen[i], os.path.join(log_eval_recon, 'img_batch{}_idx{}_gen.png'.format(batch_idx, i)))
  

                
    
    def gauss_entropies(self, pred_t, pred_var):
        entropies = np.zeros([pred_t.size()[0]])
        for i in range(0, pred_t.size()[0]):
            m = torch.distributions.MultivariateNormal(pred_t[i], torch.eye(3) * pred_var[i])
            entropies[i] = m.entropy().numpy()
        return entropies

    def bingham_entropies(self, lambdas):
        entropies = torch_bingham.bingham_entropy(lambdas)
        return entropies

    def extract_predictions(self, outputs, target, type=None, batch_idx=-1):
        batch_size = target.shape[0]
        predicted_rotation = outputs[0].reshape(-1, self.args.num_coeff, 4)
        predicted_lambda = outputs[1].reshape(-1, self.args.num_coeff, 3)
        predicted_translation = outputs[3].reshape(-1, self.args.num_coeff, 3)
        predicted_var = outputs[4].reshape(-1, self.args.num_coeff, 3)

        coeffs = outputs[2]

        predicted_ts = np.zeros([batch_size, 3])
        predicted_rs = np.zeros([batch_size, 4])
        predicted_lambdas = np.zeros([batch_size, 3])
        predicted_vars = np.zeros([batch_size, 3])
        
        idxs = []
        
        
        if type == 'highest':
            # pose from one model with largest coefficient
            index = np.asarray(np.argsort(coeffs, axis=1), dtype=np.int16)
        
            
            for i in range(0, batch_size):

                predicted_rs[i] = predicted_rotation[i, index[i, -1]]
                predicted_lambdas[i] = predicted_lambda[i, index[i, -1]]
                predicted_ts[i] = predicted_translation[i, index[i, -1]]
                predicted_vars[i] = predicted_var[i, index[i, -1]]
            
            idxs = index[:, -1]

        else:
            # Oracle prediction
            gts = np.tile(target, (self.args.num_coeff, 1, 1))
            predicted_rt = np.concatenate([predicted_rotation, predicted_translation], axis=2)

            for i in range(0, batch_size):
                best, idx = utils.close_to_all_pose(predicted_rt[i], gts[:, i, :])

                predicted_rs[i] = best[:4]
                predicted_lambdas[i] = predicted_lambda[i, idx]
                predicted_ts[i] = best[4:]
                predicted_vars[i] = predicted_var[i, idx]
                
                idxs.append(idx)
            
            idxs = np.asarray(idxs)
        
        return predicted_rs, predicted_lambdas, predicted_ts, predicted_vars, idxs

