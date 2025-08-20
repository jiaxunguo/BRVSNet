# BRVSNet

## Dependencies
The code is implemented in Python 3 (https://docs.python.org/3/license.html)

- PyTorch (BSD License: https://github.com/pytorch/pytorch/blob/master/LICENSE)

- Torchvision (BSD License: https://github.com/pytorch/vision/blob/master/LICENSE)

- Bingham Statistics Library (BSD License: https://github.com/SebastianRiedel/bingham/blob/master/LICENSE) 

- Numpy (BSD License: https://numpy.org/license.html)

- transforms3d (BSD License: https://github.com/matthew-brett/transforms3d/blob/master/LICENSE)

- PyTorch3D (BSD License: https://github.com/facebookresearch/pytorch3d/blob/master/LICENSE)

- TensorboardX (MIT License: https://github.com/lanpa/tensorboardX/blob/master/LICENSE)

## General structure

`torch_bingham/`: contains the pytorch extension for *Bingham distribution*

`cam_reloc/`: contains the code for *AmReLoc dataset --- Camera Relocalization* 

`object_pose/`: contains the code for *ModelNet10 --- Point Cloud Pose Estimation*

## Running the code

We provide main scripts for training our models for each application and a .yml file with dependencies to run our code.

### AmReLoc dataset --- Camera Relocalization and Multi-View Generation

#### Training

**Alignment Training**:

`python main.py --scene seminar --num_coeff 10 --sche_steps 50 --sche_eplr 0.7 --adam_delay 1e-4  --stage 0 --loss CE --training --num_epochs 300`

`python main.py --scene seminar --num_coeff 10 --sche_steps 50 --sche_eplr 0.7 --adam_delay 1e-4 --stage 1 --loss CE --training --restore --model model_299_0 --num_epochs 300`

**Weight Training** - To refine for rotations using Bingham Mixture Models:

`python main.py --scene seminar --num_coeff 10 --sche_steps 50 --sche_eplr 0.7 --adam_delay 1e-4 --stage 2 --loss CE --training --restore --model model_299_1 --num_epochs 300`

**Generation Training** - To refine for mixture coefficients using a cross-entropy loss:

`python main.py --scene seminar --num_coeff 10 --sche_steps 50 --sche_eplr 0.7 --adam_delay 1e-4 --stage 3 --loss CE --training --restore --model model_299_2 --num_epochs 300`

`python main.py --scene seminar --num_coeff 10 --sche_steps 50 --sche_eplr 0.7 --adam_delay 1e-4 --stage 4 --loss CE --training --restore --model model_299_3 --num_epochs 300`

#### Evaluation
To load a trained model and evaluate run:

`python main.py --restore --model "Specify your model"`

Or to check the 'log_eval.txt' files.

#### Multi-View Generation

`python eval_recon.py --scene seminar --num_coeff 10 --loss CE --restore --model model_299_4`

#### Dataset
[link](http://campar.in.tum.de/files/AmbiguousRelocDataset/Ambiguous_ReLoc_Dataset.zip)

Export to './cam_reloc/Ambiguous_ReLoc_Dataset/'.

### ModelNet10 --- Point Cloud Pose Estimation and Multi-View Generation

#### Pose Estimation

All three stages are merged in one step for this application.

`python train.py --cls_id 0 --num_model 50 --loss CE --log_dir save`

`python eval.py --num_model 50 --loss CE --log_dir save`

#### Multi-View Generation

Only one shape kept for this application in ModelNet10 because this dataset has different shapes in each category.

`python train.py --cls_id 0 --num_model 50 --loss CE --mvg_mode --log_dir save_recon`

`python eval_recon.py --cls_id 0 --num_model 50 --loss CE --log_dir save_recon`

#### Dataset
[link](https://modelnet.cs.princeton.edu/)

### Environment Setup

1. Create a new python environment. We recommend to use `conda`.
```bash
conda env create --file environment.yml
```

2. Activate the new environment
```bash
conda activate BRVSNet
```

3. Compile and install torch_bingham extension

```bash
cd torch_bingham
python setup.py build
python setup.py install
```

