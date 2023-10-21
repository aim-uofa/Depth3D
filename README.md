# Depth3D

## Install
```bash
conda create -n metricdepth python=3.7
conda activate metricdepth
pip install torch==1.10.0+cu111 torchvision==0.11.0+cu111 torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
pip install -U openmim
mim install mmengine
mim install "mmcv-full==1.3.17"
# pip install "mmsegmentation==0.19.0"
# pip install xformers==0.0.16
```

### For 40 Series GPUs
```bash
conda create -n metricdepth python=3.8
conda activate metricdepth
pip3 install torch torchvision torchaudio
pip install -r requirements.txt
pip install -U openmim
mim install mmengine
mim install "mmcv-full==1.7.1"
# pip install "mmsegmentation==0.30.0"
pip install numpy==1.20.0 # or pip install numpy==1.21.6
pip install scikit-image==0.18.0
# pip install xformers==0.0.18
```



# Dataset Components
## Data Structure
- Taskonomy
	- Taskonomy
		- rgb
		- depth
		- (optional) sem
		- (optional) normal

## Format 1
The format of annotation.json files：
```
dict(
	'files': [
		dict('meta_data': 'Taskonomy/xxx/xxx.pkl'),
		dict('meta_data': 'Taskonomy/xxx/xxx.pkl'),
		dict('meta_data': 'Taskonomy/xxx/xxx.pkl'),
		...
	]
)
```



The format of 'xxx.pkl'：
```
dict(
	'cam_in': [fx, fy, cx, cy],
	'rgb': 'Taskonomy/rgb/xxx.png',
	'depth': 'Taskonomy/depth/xxx.png',
	(optional) 'sem': 'Taskonomy/sem/xxx.png'
	(optional) 'normal': 'Taskonomy/norm/xxx.png',
)
```

## Format 2
The format of annotation.json files：
```
dict(
	'files': [
		dict('cam_in': [fx, fy, cx, cy], 'rgb': 'Taskonomy/rgb/xxx.png', 'depth': 'Taskonomy/depth/xxx.png', (optional) 'sem': 'Taskonomy/sem/xxx.png', (optional) 'normal': 'Taskonomy/norm/xxx.png'),
		dict('cam_in': [fx, fy, cx, cy], 'rgb': 'Taskonomy/rgb/xxx.png', 'depth': 'Taskonomy/depth/xxx.png', (optional) 'sem': 'Taskonomy/sem/xxx.png', (optional) 'normal': 'Taskonomy/norm/xxx.png'),
		dict('cam_in': [fx, fy, cx, cy], 'rgb': 'Taskonomy/rgb/xxx.png', 'depth': 'Taskonomy/depth/xxx.png', (optional) 'sem': 'Taskonomy/sem/xxx.png', (optional) 'normal': 'Taskonomy/norm/xxx.png'),
		...
	]
)
```


# Evaluation with Dataloader
```bash
source scripts/test/beit/test_beit_nyu.sh # change the config file if required.
```

if you would like to evaluate on a new datasets:

1. generate test annotations following ```'dataset components'```.
2. add annotation path in ```data_info/public_datasets.py```.
3. write the config file and Dataset.
4. change the config files in test_beit_nyu.sh.


# Structure of Code
```bash
- Depth3D
	- data_info
		- check_datasets.py
		- pretrained_weight.py # pretrained weight path of backbone.
		- public_datasets.py # path of annotations of diverse datasets.
	- mono
		- configs # configs of training and evaluation
		- datasets # torch.utils.data.Dataset
		- model # depth models
		- scripts # useless here
		- utils
	- scrips
	- other_tools
```

