# Basic imports
import os
import argparse
import torch
from torchvision import transforms
import numpy as np
import random
from CamPose import CamPose
import eval
from dataset_loaders.ambiguous_reloc_data import AmbiguousRelocData


seed = 13
print('Seed: %d' % seed)

np.random.seed(seed)
torch.random.manual_seed(seed)
random.seed(seed)

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--dataset', default='ReLoc', help='Specify which dataset to use, "ReLoc" for our ambiguous scene dataset.')
parser.add_argument('--scene', default='meeting_table', help='Specify which scene to use, "seminar", "meeting table", ...')
parser.add_argument('--base_dir', default='./',help='Base directory to use.')
parser.add_argument('--save_dir', default='save/', help='Directory to save models in.')
parser.add_argument('--data_dir', default='Ambiguous_ReLoc_Dataset/', help='Data directory.')
parser.add_argument('--model', default='model_299', help='Model to restore.')
parser.add_argument('--num_coeff', type=int, default=50, help='Number of components in the mixture model')
parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate for training.')
parser.add_argument('--batch_size', type=int, default=20, help='Batch Size for training.')
parser.add_argument('--num_epochs', type=int, default=300, help='Number of epochs to train.')
parser.add_argument('--stage', type=int, default=0, help='Define which stage of training to use. 0: Only translation, 1: only rotation, '
                                        'any other number: simultaneously train all distribution components.')
parser.add_argument('--restore', action='store_true', default=False, help='Set to True to restore the model specified in argument "model".')
parser.add_argument('--base', default='ResNet', help='Specifies the backbone network to use, either "ResNet" (for resnet 34)'
                                                     'or "Inception" (v3).')
parser.add_argument('--prediction_type', default='highest', help='Single best prediction according to the highest mixture coefficient.')

parser.add_argument('--sche_steps', type=int, default=50, help='Steps for schedule ExponentialLR.')
parser.add_argument('--sche_eplr', type=float, default=0.7, help='eplr for ExponentialLR.')
parser.add_argument('--adam_delay', type=float, default=1e-4, help='Weight delay of Adam.')
parser.add_argument('--loss', default='CE', help='CE or MB, loss function.')

args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = '{}'.format(args.gpu)


if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
    
#torch.backends.cudnn.deterministic = True
#torch.backends.cudnn.benchmark = False

# rotation and translation thresholds for evaluation
thresholds = [tuple([10, 0.1]), tuple([15, 0.2]), tuple([20, 0.3]), tuple([60, 1.0])]
print('Rotation')

# model
model = CamPose(args, device=device, pretrained=True)
crop_size = [224,224]
print('Model')

# image transformations
tforms = [transforms.Resize((256)),
    transforms.RandomCrop(crop_size), transforms.ToTensor()]
data_transform = transforms.Compose(tforms)
print('image transformations')

# datasets
kwargs = dict(scene=args.scene, data_path=args.data_dir, transform=data_transform)
test_tforms = [transforms.Resize((256)),
          transforms.CenterCrop(crop_size), transforms.ToTensor()]
test_data_transform = transforms.Compose(test_tforms)

print('Dataset')

kwargs_test = dict(scene=args.scene, data_path=args.data_dir, transform=test_data_transform)

if args.dataset == 'ReLoc':
    train_set = AmbiguousRelocData(train=True, **kwargs)
    test_set = AmbiguousRelocData(train=False, **kwargs_test)
else:
    raise NotImplementedError

print('Dataset loading done')

val_loader = torch.utils.data.DataLoader(test_set,
                            batch_size=args.batch_size, shuffle=False,
                            num_workers=6, pin_memory=True)

print('Loader build done')

print('Recon Evaluation')

model.eval_recon(val_loader)

print('Done')

