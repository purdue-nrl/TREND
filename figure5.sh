CUDA_VISIBLE_DEVICES=3 python trans.py --save_path=./outputs/cifar10/transferability/ --dataset=cifar10 --load_model=HT  --analysis=quant --num_steps=4 
CUDA_VISIBLE_DEVICES=3 python trans.py --save_path=./outputs/cifar10/transferability/ --dataset=cifar10 --load_model=Q1  --analysis=quant --num_steps=4 
CUDA_VISIBLE_DEVICES=3 python trans.py --save_path=./outputs/cifar10/transferability/ --dataset=cifar10 --load_model=Q2  --analysis=quant --num_steps=4 
CUDA_VISIBLE_DEVICES=3 python trans.py --save_path=./outputs/cifar10/transferability/ --dataset=cifar10 --load_model=Q4  --analysis=quant --num_steps=4 
CUDA_VISIBLE_DEVICES=3 python trans.py --save_path=./outputs/cifar10/transferability/ --dataset=cifar10 --load_model=Q6  --analysis=quant --num_steps=4 
CUDA_VISIBLE_DEVICES=3 python trans.py --save_path=./outputs/cifar10/transferability/ --dataset=cifar10 --load_model=Q8  --analysis=quant --num_steps=4 
CUDA_VISIBLE_DEVICES=3 python trans.py --save_path=./outputs/cifar10/transferability/ --dataset=cifar10 --load_model=FP  --analysis=quant --num_steps=4 
CUDA_VISIBLE_DEVICES=3 python fit_calc_TM.py  --models=HT_resnet,Q1_resnet,Q2_resnet,Q4_resnet,Q6_resnet,Q8_resnet,FP_resnet --dataset=cifar10 --save_path=./outputs/cifar10/transferability/