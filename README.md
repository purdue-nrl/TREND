# TREND: Transferability based Robust ENsemble Design

This repository is the official implementation of the paper "TREND: Transferability based Robust ENsemble Design".


## Citation
If you find our repository useful for your research, please consider citing our paper:
```bibtex
@article{ravikumar2022trend,
  title={TREND: Transferability based Robust ENsemble Design},
  author={Ravikumar, Deepak and Kodge, Sangamesh D and Garg, Isha and Roy, Kaushik},
  journal={IEEE Transactions on Artificial Intelligence}, 
  year={2023},
  volume={4},
  number={3},
  pages={534-548},
  doi={10.1109/TAI.2022.3175172}
}
```

## Requirements

**How to setup the environment?**

It is highly recommended to use Anaconda. To install Anaconda please see https://docs.anaconda.com/anaconda/install/. Once Anaconda is installed navigate to this repo and execute the following commands


```setup
conda create --name TREND python=3.6
conda activate TREND
pip install -r requirements.txt
```

## Attacking Ensembles 
To attack ensembles
```
python attack_ensemble.py --attack_type=sign_avg --dataset=cifar10 --models=FP_resnet,Q6_resnet,Q8_resnet --save_path=./outputs/cifar10/transferability/ 
```
This generates the numbers for a single ensemble, must be repeated for different ensembles

```
python attack_ensemble.py --attack_type=avg --dataset=cifar10 --models=FP_resnet,Q6_resnet,Q8_resnet --save_path=./outputs/cifar10/transferability/ 
```
This generates the numbers for a single ensemble, must be repeated for different ensembles

Attack types:
* For Direction of Average Gradient (D-AG) attack use *--attack_type=avg* 

* For Average Gradient Direction (A-GD) attack use *--attack_type=sign_avg* 

* For Unanimous Gradient Direction (U-GD) attack use *--attack_type=sign_all*

## Training
The models presented in the paper were trained and evaluated using train.py.
To train the models on ImageNet make sure to update the path in ./utils/load_dataset.py line 175 to the path where imagenet dataset is stored

```
datapath = 'Path for image net goes here' # Set path here
```

### Input Quantized Models
Here is an example on how to use train.py to train input quantized models on CIFAR10, CIFAR100 and ImageNet. 

*CIFAR10*

```train
python train.py --epochs=350 --dataset=cifar10 --load_model=Q1 --train_batch_size=32 --test_batch_size=32
```

*CIFAR100*
```train
python train.py --epochs=350 --dataset=cifar100 --load_model=Q1 --train_batch_size=32 --test_batch_size=32
```

*ImageNet*


```train
python train.py --epochs=250 --dataset=imagenet --load_model=Q1 --train_batch_size=32 --test_batch_size=32
```

where --load_model=Q\<bit-width> specifies the desired bit width, for example the commands above train Q1 model. Valid values for the "load model" arguments are FP, Q1, Q2, Q4, Q6, Q8 and HT (halftone).   

### Weight Quantized Models
The example commands below train 8 bit weight quantized models. Set --wbit argument to the desired bit width.

*CIFAR10*

```train
python train.py --epochs=350 --dataset=cifar10 --dorefa=t --wbit=8 --train_batch_size=32 --test_batch_size=32
```

*CIFAR100*
```train
python train.py --epochs=350 --dataset=cifar10 --dorefa=t --wbit=8 --train_batch_size=32 --test_batch_size=32
```

*ImageNet*


```train
python train.py --epochs=350 --dataset=imagenet --dorefa=t --wbit=8 --train_batch_size=32 --test_batch_size=32
```


### Activation Quantized Models
The example commands below train 8 bit activation quantized models. Set --abit argument to the desired bit width.

*CIFAR10*

```train
python train.py --epochs=350 --dataset=cifar10 --dorefa=t --abit=8 --train_batch_size=32 --test_batch_size=32
```

*CIFAR100*
```train
python train.py --epochs=350 --dataset=cifar10 --dorefa=t --abit=8 --train_batch_size=32 --test_batch_size=32
```

*ImageNet*


```train
python train.py --epochs=350 --dataset=imagenet --dorefa=t --abit=8 --train_batch_size=32 --test_batch_size=32
```


## Evaluating model accuracies

To evaluate the models

```eval
python train.py --epochs=0 --dataset=imagenet --load_model=FP --train_batch_size=32 --test_batch_size=32 --arch=torch_resnet18

```

```
python train.py --epochs=0 --dataset=imagenet --load_model=Q1 --train_batch_size=32 --test_batch_size=32 --arch=torch_resnet18
```

## Pre-trained Models

Pretrained models are available at https://drive.google.com/file/d/1Qyt1ZWSvatKgLFL1_gwl0AC6uyncxb5y/view

## Reproducing results from the paper

Generating numbers for Table 1 in the paper run 

```
python trans.py --limit=1000 --save_path=./ --dataset=cifar10 --analysis=seed --arch_load=resnet
python trans.py --limit=1000 --save_path=./ --dataset=cifar10 --analysis=seed --arch_load=vgg11
python trans.py --limit=1000 --save_path=./ --dataset=cifar10 --analysis=seed --arch_load=vgg11bn
```

for CIFAR10 and
```
python trans.py --limit=1000 --save_path=./ --dataset=cifar100 --analysis=seed --arch_load=resnet
python trans.py --limit=1000 --save_path=./ --dataset=cifar100 --analysis=seed --arch_load=vgg11
python trans.py --limit=1000 --save_path=./ --dataset=cifar100 --analysis=seed --arch_load=vgg11bn
```

for CIFAR100

### Figure 2a
To generate numbers for Figure 1a's first row run, --load_model specifes the source
```
python trans.py --limit=1000 --save_path=./ --dataset=cifar10 --analysis=arch --arch_load=resnet
```
Similarly for the second row
```
python trans.py --limit=1000 --save_path=./ --dataset=cifar10 --analysis=arch --arch_load=resnet34
```
### Figure 2b
To generate numbers for Figure 1b's first row run, --arch_load specifes the source model architecture, --limit specifies the  subset size, torch_weights uses pytorch trained weights
```
python trans.py --limit=1000 --save_path=./ --dataset=imagenet --analysis=arch --torch_weights=T --arch_load=torch_resnet18
```
Similarly for the second row
```
python trans.py --limit=1000 --save_path=./ --dataset=imagenet --analysis=arch --torch_weights=T --arch_load=torch_resnet34
```
and so on..

### Figure 3a
To generate numbers for Figure 2a's first row run, --load_model specifes the source
```
python trans.py --limit=1000 --save_path=./ --dataset=cifar10 --analysis=quant --load_model=HT
```
Similarly for the second row
```
python trans.py --limit=1000 --save_path=./ --dataset=cifar10 --analysis=quant --load_model=Q1
```

### Figure 3b
To generate numbers for Figure 2b's first-row run, --load_model specifies the source
```
python trans.py --limit=1000 --save_path=./ --dataset=imagenet --analysis=quant --load_model=HT
```
Similarly for the second row
```
python trans.py --limit=1000 --save_path=./ --dataset=imagenet --analysis=quant --load_model=Q1
```

### Figure 4a
To generate numbers for Figure 3a's first row run
```
python trans.py --limit=1000 --save_path=./ --dataset=cifar10 --analysis=weight --dorefa_load=T  --wbit_load=1
```
Similarly for the second row
```
python trans.py --limit=1000 --save_path=./ --dataset=cifar10 --analysis=weight  --dorefa_load=T  --wbit_load=2
```

### Figure 4b
To generate numbers for Figure 3b's first row run
```
python trans.py --limit=1000 --save_path=./ --dataset=cifar10 --analysis=act --dorefa_load=T  --abit_load=1
```
