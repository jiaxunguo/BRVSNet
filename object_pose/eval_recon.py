import argparse

import os
import numpy as np
import torch
import network_bingham
from utils import qrotate_pc, qconjugate, qmult
from pytorch3d.loss import chamfer_distance
from data_modelnet import load_testdata, load_sameobject_test_modelnet10

from mayavi import mlab

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--log_dir', default='save_test', help='Log dir [default: log]')
parser.add_argument('--num_point', type=int, default=2048, help='Point number [default: 2048]')
parser.add_argument('--num_model', type=int, default=50, help='Number of models used')
parser.add_argument('--max_epoch', type=int, default=50000, help='Epoch to run [default: 50000]')
# parser.add_argument('--batch_size', type=int, default=32, help='Batch Size during training [default: 32]')
# parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--cls_id', type=int, default=0, help='The id of the target object class')
parser.add_argument('--weight_fn', default=None, help='Pre-trained weights for the network')
parser.add_argument('--loss', default='CE', help='CE or MB, loss function.')
parser.add_argument('--use_l1', action='store_true', default=False, help='Set to True to enable l1 in loss.')
parser.add_argument('--config', default='oim10', help='oim06, oim04, oim08, or oim10, rotation_samples.')

m10_clsses = ['bathtub', 'bed', 'chair', 'desk', 'dresser', 'monitor',
              'night_stand', 'sofa', 'table', 'toilet']

flags = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = '{}'.format(flags.gpu)
cls = m10_clsses[flags.cls_id]


num_point = flags.num_point
# batch_size = flags.batch_size
device = torch.device('cuda:0')

if flags.loss == 'CE':
    loss_str = 'CE'
else:
    loss_str = 'MB'

nm = flags.num_model

net = network_bingham.MBN(num_point, 3, 128, nm, flags.loss)
net = net.to(device)

mlab.options.offscreen = True
fig = mlab.figure(size=(400, 400), bgcolor=(1,1,1))


def eval_cls(cls):
    ## change the weight_fn to the expected one
    weight_fn = flags.log_dir + '/log_{}/numcoeff{}_{}/chkpt.pth'.format(cls, str(flags.num_model), loss_str)
    
    save_dir = flags.log_dir + '/log_{}/numcoeff{}_{}/'.format(cls, str(flags.num_model), loss_str)
    recon_dir = os.path.join(save_dir, 'test_recon')
    if not os.path.exists(recon_dir):
        os.makedirs(recon_dir)
        
    reconTxt_dir = os.path.join(recon_dir, 'txt')
    if not os.path.exists(reconTxt_dir):
        os.makedirs(reconTxt_dir)
        
    if not os.path.exists(weight_fn):
        print('{} not exists.'.format(weight_fn))
        return
    

    print('Initializing network')

    state_dict = torch.load(weight_fn)
    print('loading weights from {}'.format(weight_fn))
    net.load_state_dict(state_dict, strict=False)
    net.eval()
    print('Network initialization done')
    
    test_data = load_sameobject_test_modelnet10(cls, config=flags.config)

    for batch_idx, (pc, gt_q) in enumerate(test_data):
        points = torch.from_numpy(pc).float().to(device).reshape(1, num_point, pc.shape[1])
        gt_q = torch.from_numpy(gt_q).float().to(device).reshape(1, 4)

        # print(batch_idx, ':', gt_q)
        x_gen = net.recon(gt_q)
        
        for i in range(points.shape[0]):
            # mayavi      
            point_recon_org = points[i].detach().cpu().numpy()
            point_recon_sig_gen = x_gen[i].detach().cpu().numpy()
            
            np.savetxt(os.path.join(reconTxt_dir, 'point_batch{}_idx{}_org.txt'.format(batch_idx, i)), point_recon_org, delimiter=',', fmt='%.5f')
            np.savetxt(os.path.join(reconTxt_dir, 'point_batch{}_idx{}_sig_gen.txt'.format(batch_idx, i)), point_recon_sig_gen, delimiter=',', fmt='%.5f')
            
            mlab.points3d(point_recon_org[:,0], point_recon_org[:,1], point_recon_org[:,2], color=(0,0,1), scale_factor=0.05)
            mlab.savefig(os.path.join(recon_dir, 'img_batch{}_idx{}_org.png'.format(batch_idx, i)))
            mlab.clf(fig)
            
            mlab.points3d(point_recon_sig_gen[:,0], point_recon_sig_gen[:,1], point_recon_sig_gen[:,2], color=(0,0,1), scale_factor=0.05)
            mlab.savefig(os.path.join(recon_dir, 'img_batch{}_idx{}_sig_gen.png'.format(batch_idx, i)))
            mlab.clf(fig)
    

if __name__ == '__main__':
    eval_cls(cls)
    mlab.close(fig)
