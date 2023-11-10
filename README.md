# Depth3D

## Install
```bash
conda create -n Depth3D python=3.7
conda activate Depth3D
pip install torch==1.10.0+cu111 torchvision==0.11.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
pip install -U openmim
mim install mmengine
mim install "mmcv-full==1.3.17"
pip install yapf==0.40.1
```

### For 40 Series GPUs
```bash
conda create -n Depth3D python=3.8
conda activate Depth3D
pip install torch==2.0.0 torchvision==0.15.1
pip install -r requirements.txt
pip install -U openmim
mim install mmengine
mim install "mmcv-full==1.7.1"
pip install yapf==0.40.1
```



# Dataset Components
We offer the test split of datasets with the Baidu Netdisk share link.

## Data Structure
We use the *_annotation.json files to store the camera intrinsic information and the paths of rgb, depth, etc. The data structure is as follows:

```
- Taskonomy
	- Taskonomy
		- (optional) meta # save the pickle files, see 'Format 1' for details
		- rgb
		- depth
		- (optional) sem
		- (optional) normal
	- test_annotation.json # test annotation file
	- train_annotation.json # train annotation file
```
## Format 1
The format of *_annotation.json files：
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
The format of *_annotation.json files：
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

# Reproducing the Results of the Report
Run scripts of ```Depth3D/scripts/technical_report```. 
For example. if you would like to reproduce the Table 3, run this commmand: ```python scripts/technical_report/run_table3.py```. It will take hours of time to output the final results.

See ```scripts/technical_report/README.md``` for details

# Evaluation of a Specific Dataset with Dataloader
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
	- datasets
		- ibims
			- ibims
			- test_annotation.json
		- diode
			- diode
			- test_annotation.json
			- test_annotation_indoor.json
			- test_annotation_outdoor.json
		- ETH3D
			- ETH3D
			- test_annotations.json
		...
	- demo_data # demo data of technical report.
	- mono
		- configs # configs of training and evaluation.
		- datasets # torch.utils.data.Dataset.
		- model # depth models.
		- tools
		- utils
	- other_tools
	- pretrained_weights # pretrained weights, used for training depth models.
	- scripts # scripts of training and testing
		- ablation 
			- test
			- train
		- technical_report # scripts to reproduce the results of technical report.
		- test
		- train
	- show_dirs # output folder of inference.
	- weights # place our trained depth models here.
	- weights_ablation # place our released depth models of ablation study here.
	- work_dirs # output folder of training.
```

