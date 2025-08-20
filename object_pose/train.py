import os
import argparse
import sys

import numpy as np

import torch
from torch.nn import functional as F
from tensorboardX import SummaryWriter
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR

import loss_functions
import network_bingham
from data_modelnet import load_modelnet10, PointDownsample, ModelNetDatasetIcoshpere, load_sameobject_modelnet10

from utils import qrotate_pc, qconjugate, qmult
from pytorch3d.loss import chamfer_distance

from mayavi import mlab

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--log_dir', default='save_dec', help='Log dir [default: log]')
parser.add_argument('--num_point', type=int, default=2048, help='Point number [default: 2048]')
parser.add_argument('--num_model', type=int, default=10, help='Number of models used')
parser.add_argument('--max_epoch', type=int, default=50000, help='Epoch to run [default: 50000]')
parser.add_argument('--batch_size', type=int, default=32, help='Batch Size during training [default: 32]')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--cls_id', type=int, default=0, help='The id of the target object class')
parser.add_argument('--weight_fn', default=None, help='Pre-trained weights for the network')
parser.add_argument('--loss', default='CE', help='CE or MB, loss function.')
parser.add_argument('--use_l1', action='store_true', default=False, help='Set to True to enable l1 in loss.')
parser.add_argument('--mvg_mode', action='store_true', default=False, help='Set to True to multi-view generation mode.')

m10_clsses = ['bathtub', 'bed', 'chair', 'desk', 'dresser', 'monitor',
              'night_stand', 'sofa', 'table', 'toilet']

flags = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = '{}'.format(flags.gpu)
cls = m10_clsses[flags.cls_id]
num_point = flags.num_point
nm = flags.num_model
max_epoch = flags.max_epoch
batch_size = flags.batch_size
learning_rate = flags.learning_rate

if flags.loss == 'CE':
    loss_str = 'CE'
else:
    loss_str = 'MB'


save_dir = flags.log_dir + '/log_{}/numcoeff{}_{}/'.format(cls, str(flags.num_model), loss_str)
print('save dir: {}'.format(save_dir))

device = torch.device('cuda:0')

weight_fn = None
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
    
        
recon_dir = os.path.join(save_dir, 'recon')
if not os.path.exists(recon_dir):
    os.makedirs(recon_dir)
    
reconTxt_dir = os.path.join(recon_dir, 'txt')
if not os.path.exists(reconTxt_dir):
    os.makedirs(reconTxt_dir)

log_fn = os.path.join(save_dir, 'log.txt')
fid = open(log_fn, 'w')
summary_writer = SummaryWriter(os.path.join(save_dir, 'summary'))


if flags.loss == 'CE':
    criterion = loss_functions.RWTA_CE_KL_Loss(nm)    
else:
    criterion = loss_functions.RWTA_MB_KL_Loss(nm)


mlab.options.offscreen = True
fig = mlab.figure(size=(400, 400), bgcolor=(1,1,1))

def run_epoch_gen(net, dataloader, optimizer, epoch, is_training=True):
    if is_training:
        net.train()
    else:
        net.eval()
        
    rwta_loss_lst = []
    mb_loss_lst = []
    cd_loss_lst = []
    cd_sig_loss_lst = []
    loss_lst = []

    n_batch = len(dataloader)
    
    cd_lst = []
    for batch_idx, (data, label) in enumerate(dataloader):
        if is_training:
            net.zero_grad()

        points = data.float().to(device)
        label = label.float().to(device)
        label = ((label[:, 0:1] > 0).float() - 0.5) * 2 * label   
        
        pred_q, pred_l, weights, x_dis, x_gen = net(points)
        rwta_loss, mb_loss = criterion(pred_q, pred_l, weights, label)
        
        cd_loss = chamfer_distance(points.reshape(points.shape[0], num_point, 3), x_dis.reshape(x_dis.shape[0], num_point, 3), batch_reduction='mean')[0]
        cd_loss = 100 * cd_loss    
                
        cd_sig_loss = chamfer_distance(points.reshape(points.shape[0], num_point, 3), x_gen.reshape(x_gen.shape[0], num_point, 3), batch_reduction='mean')[0]
        cd_sig_loss = 100 * cd_sig_loss
        
        loss = 1.0 * rwta_loss + mb_loss + cd_loss + cd_sig_loss
        
        rwta_loss_lst.append(rwta_loss.item())
        mb_loss_lst.append(mb_loss.item())
        cd_loss_lst.append(cd_loss.item())
        cd_sig_loss_lst.append(cd_sig_loss.item())
        loss_lst.append(loss.item())

        if is_training:
            loss.backward()
            optimizer.step()
            
            cd_lst.append(-1)
        else:
            for idx, (pc, gt_q) in enumerate(zip(data, label)):
                point = pc.to(device).reshape(1, num_point, pc.shape[1])
                gt_q = gt_q.to(device).reshape(1, 4)
                            
                rel_q = qmult(pred_q[idx*nm:(idx+1)*nm], qconjugate(gt_q))

                rel_q_tiled = rel_q.reshape(nm, 1, 4).repeat(1, pc.shape[0], 1).reshape(-1, 4)
                points_tiled = point.reshape(1, pc.shape[0], 3).repeat(nm, 1, 1).reshape(-1, 3)
            
                rotated_pc = qrotate_pc(points_tiled, rel_q_tiled)
                rotated_pc = rotated_pc.reshape(nm, pc.shape[0], 3)

                dists = chamfer_distance(points_tiled.reshape(nm, pc.shape[0], 3), rotated_pc, batch_reduction=None)[0]
                best_dist = dists[weights[idx].argmax()].item()

                cd_lst.append(best_dist)

        print("{}/{}".format(batch_idx, n_batch), end='\r')
    
    mean_rwta_loss = np.mean(rwta_loss_lst)
    mean_mb_loss = np.mean(mb_loss_lst)
    mean_cd_loss = np.mean(cd_loss_lst)
    mean_cd_sig_loss = np.mean(cd_sig_loss_lst)
    mean_loss = np.mean(loss_lst)
    
    print_str = "Epoch: {} Loss:{:.4f} RWTA:{:.4f} MB:{:.4f} CD_dis:{:.4f} CD_gen:{:.4f}".format(epoch, mean_loss, mean_rwta_loss, mean_mb_loss, mean_cd_loss, mean_cd_sig_loss)    
    
    if is_training:
        if (epoch + 1) % 100 == 0:
            print(print_str)
        fid.write(print_str + '\n')
        
        summary_writer.add_scalar('train/loss', mean_loss, epoch)
        summary_writer.add_scalar('train/rwta_loss', mean_rwta_loss, epoch)
        summary_writer.add_scalar('train/mb_loss', mean_mb_loss, epoch) 
        summary_writer.add_scalar('train/cd_dis_loss', mean_cd_loss, epoch)         
        summary_writer.add_scalar('train/cd_gen_loss', mean_cd_sig_loss, epoch)
    else:
        print('Eval ' + print_str)
        fid.write('Eval ' + print_str + '\n')
        print('chamfer_distance_val: {}'.format(np.mean(cd_lst)))
        fid.write('chamfer_distance_val: {} \n'.format(np.mean(cd_lst)))
        
        summary_writer.add_scalar('eval/loss', mean_loss, epoch)
        summary_writer.add_scalar('eval/rwta_loss', mean_rwta_loss, epoch)
        summary_writer.add_scalar('eval/mb_loss', mean_mb_loss, epoch)
        summary_writer.add_scalar('eval/CD_dis_loss', mean_cd_loss, epoch)        
        summary_writer.add_scalar('eval/cd_gen_loss', mean_cd_sig_loss, epoch)
        summary_writer.add_scalar('eval/chamfer_distance_val', np.mean(cd_lst), epoch)
        
        # mayavi      
        point_recon_org = points[0].detach().cpu().numpy()
        point_recon_gen = x_dis[0].detach().cpu().numpy()
        point_recon_sig_gen = x_gen[0].detach().cpu().numpy()
        
        np.savetxt(os.path.join(reconTxt_dir, 'point_epoch{}_org.txt'.format(epoch)), point_recon_org, delimiter=',', fmt='%.5f')
        np.savetxt(os.path.join(reconTxt_dir, 'point_epoch{}_gen.txt'.format(epoch)), point_recon_gen, delimiter=',', fmt='%.5f')
        np.savetxt(os.path.join(reconTxt_dir, 'point_epoch{}_sig_gen.txt'.format(epoch)), point_recon_sig_gen, delimiter=',', fmt='%.5f')
        
        mlab.points3d(point_recon_org[:,0], point_recon_org[:,1], point_recon_org[:,2], color=(0,0,1), scale_factor=0.05)
        mlab.savefig(os.path.join(recon_dir, 'img_epoch{}_org.png'.format(epoch)))
        mlab.clf(fig)
        
        mlab.points3d(point_recon_gen[:,0], point_recon_gen[:,1], point_recon_gen[:,2], color=(0,0,1), scale_factor=0.05)
        mlab.savefig(os.path.join(recon_dir, 'img_epoch{}_gen.png'.format(epoch)))
        mlab.clf(fig)
        
        mlab.points3d(point_recon_sig_gen[:,0], point_recon_sig_gen[:,1], point_recon_sig_gen[:,2], color=(0,0,1), scale_factor=0.05)
        mlab.savefig(os.path.join(recon_dir, 'img_epoch{}_sig_gen.png'.format(epoch)))
        mlab.clf(fig)
              
    return mean_loss, np.mean(cd_lst)

def run_epoch_reg(net, dataloader, optimizer, epoch, is_training=True):
    if is_training:
        net.train()
    else:
        net.eval()
        
    rwta_loss_lst = []
    mb_loss_lst = []
    cd_loss_lst = []
    loss_lst = []

    n_batch = len(dataloader)
    
    cd_lst = []
    for batch_idx, (data, label) in enumerate(dataloader):
        if is_training:
            net.zero_grad()

        points = data.float().to(device)
        label = label.float().to(device)
        label = ((label[:, 0:1] > 0).float() - 0.5) * 2 * label   
        
        pred_q, pred_l, weights, x_dis, _ = net(points)
        rwta_loss, mb_loss = criterion(pred_q, pred_l, weights, label)
        
        cd_loss = chamfer_distance(points.reshape(points.shape[0], num_point, 3), x_dis.reshape(x_dis.shape[0], num_point, 3), batch_reduction='mean')[0]
        cd_loss = 100 * cd_loss    
        
        loss = 1.0 * rwta_loss + mb_loss + cd_loss
        
        rwta_loss_lst.append(rwta_loss.item())
        mb_loss_lst.append(mb_loss.item())
        cd_loss_lst.append(cd_loss.item())
        loss_lst.append(loss.item())

        if is_training:
            loss.backward()
            optimizer.step()
            
            cd_lst.append(-1)
        else:
            for idx, (pc, gt_q) in enumerate(zip(data, label)):
                point = pc.to(device).reshape(1, num_point, pc.shape[1])
                gt_q = gt_q.to(device).reshape(1, 4)
                            
                rel_q = qmult(pred_q[idx*nm:(idx+1)*nm], qconjugate(gt_q))

                rel_q_tiled = rel_q.reshape(nm, 1, 4).repeat(1, pc.shape[0], 1).reshape(-1, 4)
                points_tiled = point.reshape(1, pc.shape[0], 3).repeat(nm, 1, 1).reshape(-1, 3)
            
                rotated_pc = qrotate_pc(points_tiled, rel_q_tiled)
                rotated_pc = rotated_pc.reshape(nm, pc.shape[0], 3)

                dists = chamfer_distance(points_tiled.reshape(nm, pc.shape[0], 3), rotated_pc, batch_reduction=None)[0]
                best_dist = dists[weights[idx].argmax()].item()

                cd_lst.append(best_dist)

        print("{}/{}".format(batch_idx, n_batch), end='\r')
    
    mean_rwta_loss = np.mean(rwta_loss_lst)
    mean_mb_loss = np.mean(mb_loss_lst)
    mean_cd_loss = np.mean(cd_loss_lst)
    mean_loss = np.mean(loss_lst)
    
    print_str = "Epoch: {} Loss:{:.4f} RWTA:{:.4f} MB:{:.4f} CD_dis:{:.4f}".format(epoch, mean_loss, mean_rwta_loss, mean_mb_loss, mean_cd_loss)    
    
    if is_training:
        if (epoch + 1) % 100 == 0:
            print(print_str)
        fid.write(print_str + '\n')
        
        summary_writer.add_scalar('train/loss', mean_loss, epoch)
        summary_writer.add_scalar('train/rwta_loss', mean_rwta_loss, epoch)
        summary_writer.add_scalar('train/mb_loss', mean_mb_loss, epoch) 
        summary_writer.add_scalar('train/cd_dis_loss', mean_cd_loss, epoch)
    else:
        print('Eval ' + print_str)
        fid.write('Eval ' + print_str + '\n')
        print('chamfer_distance_val: {}'.format(np.mean(cd_lst)))
        fid.write('chamfer_distance_val: {} \n'.format(np.mean(cd_lst)))
        
        summary_writer.add_scalar('eval/loss', mean_loss, epoch)
        summary_writer.add_scalar('eval/rwta_loss', mean_rwta_loss, epoch)
        summary_writer.add_scalar('eval/mb_loss', mean_mb_loss, epoch)
        summary_writer.add_scalar('eval/CD_dis_loss', mean_cd_loss, epoch)
        summary_writer.add_scalar('eval/chamfer_distance_val', np.mean(cd_lst), epoch)
              
    return mean_loss, np.mean(cd_lst)
    
def main():
    from torch.utils.data import DataLoader
    from torchvision import transforms
    
    if flags.mvg_mode:
        data = load_sameobject_modelnet10(cls, 'train')[:, :, :3]
    else:
        data = load_modelnet10(cls, 'train')[:, :, :3]
    
    from sklearn.model_selection import train_test_split

    train_data, val_data = train_test_split(data, test_size=0.2)
    
    # test_data_fn = './data/benchmark/{}.npy'.format(cls)
    # test_data = np.load(test_data_fn, allow_pickle=True)
    
    print('train_data: {} val_data: {}'.format(train_data.shape, val_data.shape))
    
    m_transforms = transforms.Compose([
        PointDownsample(num_point)
    ])

    train_ds = ModelNetDatasetIcoshpere(train_data, m_transforms)
    val_ds = ModelNetDatasetIcoshpere(val_data, m_transforms)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, pin_memory=True)

    net = network_bingham.MBN(num_point, 3, 128, nm, flags.loss)
    net = net.to(device)
    
    
    to_optimize = net.parameters()
    
    if flags.mvg_mode:
        run_one_epoch = run_epoch_gen
    else:
        run_one_epoch = run_epoch_reg
    
    weight_decay = 1e-5
    patience=10
    factor=0.5
    min_lr=1e-5
    
    optimizer = Adam(to_optimize, lr=learning_rate, weight_decay=weight_decay)

    lr_scheduler = ReduceLROnPlateau(optimizer, patience=patience, factor=factor, min_lr=min_lr)
    if weight_fn is not None:
        print('Loading MODEL from {}'.format(weight_fn))
        state_dict = torch.load(weight_fn)
        net.load_state_dict(state_dict, strict=False)
        with torch.no_grad():
            run_one_epoch(net, val_dl, None, -1, is_training=False)

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    cd_best = np.inf
    for epoch in range(max_epoch):
        run_one_epoch(net, train_dl, optimizer, epoch, is_training=True)

        if (epoch + 1) % 100 == 0:
            with torch.no_grad():
                val_loss, cd_mean = run_one_epoch(net, val_dl, None, epoch, is_training=False)
                lr_scheduler.step(val_loss)
                
                # cd_mean = eval(test_data, net)
                
            if cd_mean < cd_best:
                cd_best = cd_mean
                ## save weights
                save_state_fn = os.path.join(save_dir, 'chkpt.pth')
                torch.save(net.state_dict(), save_state_fn)
                
                # Save log
                print('Best state save epoch: {}'.format(epoch + 1))
                fid.write('Best state save epoch: {} \n'.format(epoch + 1))

    ## save weights
    save_state_fn = os.path.join(save_dir, 'chkpt_g.pth')
    torch.save(net.state_dict(), save_state_fn)
    
    ## close file handler
    fid.close()
    mlab.close(fig)

    print('Training for {} is finished'.format(cls))


if __name__ == '__main__':
    main()
