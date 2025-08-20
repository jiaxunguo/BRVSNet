import os
import numpy as np
from transforms3d import quaternions as tq
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader


def load_modelnet10(cls='table', split='train'):
    data_fn = './data/modelnet_{}_{}.npz'.format(cls, split)
    if os.path.exists(data_fn):
        print('loading data from {}'.format(data_fn))
        return np.load(data_fn, allow_pickle=True)['pc']
    else:
        print('{} does not exist.'.format(data_fn))
        return None

def load_sameobject_modelnet10(cls='table', split='train'):
    data_fn = './data/sameobject/modelnet_{}_{}.npz'.format(cls, split)
    if os.path.exists(data_fn):
        print('loading data from {}'.format(data_fn))
        return np.load(data_fn)['pc']
    else:
        print('{} does not exist.'.format(data_fn))
        return None

def load_sameobject_test_modelnet10(cls='bathtub', config='oim10'):
    split = 'train'
    data_fn = './data/sameobject/modelnet_{}_{}.npz'.format(cls, split)
    if os.path.exists(data_fn):
        print('loading data from {}'.format(data_fn))
        data = np.load(data_fn)['pc']
    else:
        print('{} does not exist.'.format(data_fn))
        raise ValueError
    
    print('Generating testdata of {}, config: {} '.format(cls, config))
    
    m_transforms = transforms.Compose([
        PointDownsample(2048)
    ])
    
    test_ds = ModelNetDatasetIcoshpereTest(data, m_transforms, config=config)
    test_dl = DataLoader(test_ds, batch_size=1, pin_memory=True)
    
    test_n = []

    for batch_idx, (data, label) in enumerate(test_dl):
        test_n.append((data.squeeze().numpy(), label.squeeze().numpy()))
    
    test_n = np.array(test_n, dtype=object)
    
    return test_n

class PointDownsample(object):
    def __init__(self, npoints=2048):
        self.npoints = npoints

    def __call__(self, points):
        return points[np.random.choice(len(points), self.npoints)]


class ModelNetDatasetIcoshpereTest(Dataset):
    def __init__(self, data, transforms, config='oim06'):
        self.data = data
        self.len = data.shape[0]
        self.transforms = transforms
        self.quats = load_sample_poses(config)
        self.n_q = len(self.quats)
        print('Generating {} quats from {}'.format(self.n_q, config))

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        pc = self.data[idx]
        if self.transforms is not None:
            pc = self.transforms(pc)

        # r_idx = np.random.randint(0, self.n_q)
        r_idx = idx
        q = self.quats[r_idx]
        
        # print('idx:', idx)
        # print('quat: ', q)
        
        q = (2 * (q[0] > 0) - 1) * q
        # print('q: ', q)
        
        rot = tq.quat2mat(q)
        if pc.shape[1] == 3:
            t_pc = pc @ rot.T
        elif pc.shape[1] == 6:
            t_ver = pc[:, :3] @ rot.T
            t_nor = pc[:, 3:6] @ rot.T
            t_pc = np.concatenate([t_ver, t_nor], axis=1)
        return t_pc, q
    
class ModelNetDatasetIcoshpere(Dataset):
    def __init__(self, data, transforms, config='oim06'):
        self.data = data
        self.len = data.shape[0]
        self.transforms = transforms
        self.quats = load_sample_poses(config)
        self.n_q = len(self.quats)
        print('Generating {} quats'.format(self.n_q))

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        pc = self.data[idx]
        if self.transforms is not None:
            pc = self.transforms(pc)

        r_idx = np.random.randint(0, self.n_q)
        q = self.quats[r_idx]
        q = (2 * (q[0] > 0) - 1) * q
        rot = tq.quat2mat(q)
        if pc.shape[1] == 3:
            t_pc = pc @ rot.T
        elif pc.shape[1] == 6:
            t_ver = pc[:, :3] @ rot.T
            t_nor = pc[:, 3:6] @ rot.T
            t_pc = np.concatenate([t_ver, t_nor], axis=1)
        return t_pc, q


def init_rot_mat(config_fn):
    data = np.loadtxt(config_fn)
    phi = data[:, 0]
    theta = data[:, 1]
    psi = data[:, 2]
    cpsi = np.cos(psi)
    spsi = np.sin(psi)

    ctheta = np.cos(theta)
    stheta = np.sin(theta)

    cphi = np.cos(phi)
    sphi = np.sin(phi)

    rMat = np.zeros((data.shape[0], 9))

    rMat[:, 0] = cpsi * cphi - spsi * ctheta * sphi
    rMat[:, 1] = -cpsi * sphi - spsi * ctheta * cphi
    rMat[:, 2] = spsi * stheta

    rMat[:, 3] = spsi * cphi + cpsi * ctheta * sphi
    rMat[:, 4] = -spsi * sphi + cpsi * ctheta * cphi
    rMat[:, 5] = -cpsi * stheta

    rMat[:, 6] = stheta * sphi
    rMat[:, 7] = stheta * cphi
    rMat[:, 8] = ctheta

    rMat = rMat.reshape(-1, 3, 3)

    q_list = []
    for m in rMat:
        q = tq.mat2quat(m)
        q = (2 * (q[0] > 0) - 1) * q
        q_list.append(q)
    q_list = np.stack(q_list)
    return q_list


def load_sample_poses(config='oim06'):
    config_fn = './rotation_samples/{}.eul'.format(config)
    q_list = init_rot_mat(config_fn)
    return q_list
