import argparse

import os
import numpy as np
import torch
import network_bingham
from utils import qrotate_pc, qconjugate, qmult
from pytorch3d.loss import chamfer_distance

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--log_dir', default='save_test', help='Log dir [default: log]')
parser.add_argument('--num_point', type=int, default=2048, help='Point number [default: 2048]')
parser.add_argument('--num_model', type=int, default=50, help='Number of models used')
parser.add_argument('--max_epoch', type=int, default=50000, help='Epoch to run [default: 50000]')
parser.add_argument('--weight_fn', default=None, help='Pre-trained weights for the network')
parser.add_argument('--loss', default='CE', help='CE or MB, loss function.')
parser.add_argument('--use_l1', action='store_true', default=False, help='Set to True to enable l1 in loss.')
parser.add_argument('--stage', type=int, default=0, help='The stage of training.')
parser.add_argument('--config', default='oim06', help='oim06, oim04, oim08, or oim10, rotation_samples.')

m10_clsses = ['bathtub', 'bed', 'chair', 'desk', 'dresser', 'monitor',
              'night_stand', 'sofa', 'table', 'toilet']

flags = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = '{}'.format(flags.gpu)
# cls = m10_clsses[flags.cls_id]


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


def eval_cls(cls):
    ## change the weight_fn to the expected one
    weight_fn = flags.log_dir + '/log_{}/numcoeff{}_{}/chkpt.pth'.format(cls, str(flags.num_model), loss_str)
    
    if not os.path.exists(weight_fn):
        print('{} not exists.'.format(weight_fn))
        return

    print('Initializing network')

    state_dict = torch.load(weight_fn)
    print('loading weights from {}'.format(weight_fn))
    net.load_state_dict(state_dict, strict=False)
    net.eval()
    print('Network initialization done')
    
    test_data_fn = './data/modelnet_{}_test.npy'.format(cls)
    test_data = np.load(test_data_fn, allow_pickle=True)

    cd_lst = []
    for idx, (pc, gt_q) in enumerate(test_data):
        points = torch.from_numpy(pc).float().to(device).reshape(1, num_point, pc.shape[1])
        gt_q = torch.from_numpy(gt_q).float().to(device).reshape(1, 4)

        pred_q, pred_l, weights, _, _ = net(points)

        rel_q = qmult(pred_q, qconjugate(gt_q))

        rel_q_tiled = rel_q.reshape(nm, 1, 4).repeat(1, pc.shape[0], 1).reshape(-1, 4)
        points_tiled = points.reshape(1, pc.shape[0], 3).repeat(nm, 1, 1).reshape(-1, 3)

        rotated_pc = qrotate_pc(points_tiled, rel_q_tiled)
        rotated_pc = rotated_pc.reshape(nm, pc.shape[0], 3)

        dists = chamfer_distance(points_tiled.reshape(nm, pc.shape[0], 3), rotated_pc, batch_reduction=None)[0]
        best_dist = dists[weights.argmax()].item()

        cd_lst.append(best_dist)

    print('{}: {}'.format(cls, np.mean(cd_lst)))


def eval_all():
    classes = [
        'bathtub',
        # 'bed',
        # 'chair',
        # 'desk',
        # 'dresser',
        # 'monitor',
        # 'night_stand',
        # 'sofa',
        # 'table',
        # 'toilet',
    ]

    for cls in classes:
        eval_cls(cls)


if __name__ == '__main__':
    eval_all()
