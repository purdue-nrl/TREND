cd ..
CUDA_VISIBLE_DEVICES=0 python trans.py  --dataset=cifar100 --save_path=./outputs/ --use_lib=advertorch --analysis=quant --arch=resnet --attack=cw --iteration=100 --conf=30 --suffix_load="_1" --load_model=FP --test_batch_size=128
CUDA_VISIBLE_DEVICES=0 python trans.py  --dataset=cifar100 --save_path=./outputs/ --use_lib=advertorch --analysis=quant --arch=resnet --attack=cw --iteration=100 --conf=30 --suffix_load="_1" --load_model=Q1 --test_batch_size=128
CUDA_VISIBLE_DEVICES=0 python trans.py  --dataset=cifar100 --save_path=./outputs/ --use_lib=advertorch --analysis=quant --arch=resnet --attack=cw --iteration=100 --conf=30 --suffix_load="_1" --load_model=Q2 --test_batch_size=128
CUDA_VISIBLE_DEVICES=0 python trans.py  --dataset=cifar100 --save_path=./outputs/ --use_lib=advertorch --analysis=quant --arch=resnet --attack=cw --iteration=100 --conf=30 --suffix_load="_1" --load_model=Q4 --test_batch_size=128
CUDA_VISIBLE_DEVICES=0 python trans.py  --dataset=cifar100 --save_path=./outputs/ --use_lib=advertorch --analysis=quant --arch=resnet --attack=cw --iteration=100 --conf=30 --suffix_load="_1" --load_model=Q6 --test_batch_size=128
CUDA_VISIBLE_DEVICES=0 python trans.py  --dataset=cifar100 --save_path=./outputs/ --use_lib=advertorch --analysis=quant --arch=resnet --attack=cw --iteration=100 --conf=30 --suffix_load="_1" --load_model=Q8 --test_batch_size=128
CUDA_VISIBLE_DEVICES=0 python trans.py  --dataset=cifar100 --save_path=./outputs/ --use_lib=advertorch --analysis=quant --arch=resnet --attack=cw --iteration=100 --conf=30 --suffix_load="_1" --load_model=HT --test_batch_size=128
CUDA_VISIBLE_DEVICES=0 python trans.py  --dataset=cifar100 --save_path=./outputs/ --use_lib=advertorch --analysis=quant --arch=resnet --attack=cw --iteration=100 --conf=30 --suffix_load="_2" --load_model=FP --test_batch_size=128
CUDA_VISIBLE_DEVICES=0 python trans.py  --dataset=cifar100 --save_path=./outputs/ --use_lib=advertorch --analysis=quant --arch=resnet --attack=cw --iteration=100 --conf=30 --suffix_load="_2" --load_model=Q1 --test_batch_size=128
CUDA_VISIBLE_DEVICES=0 python trans.py  --dataset=cifar100 --save_path=./outputs/ --use_lib=advertorch --analysis=quant --arch=resnet --attack=cw --iteration=100 --conf=30 --suffix_load="_2" --load_model=Q2 --test_batch_size=128
CUDA_VISIBLE_DEVICES=0 python trans.py  --dataset=cifar100 --save_path=./outputs/ --use_lib=advertorch --analysis=quant --arch=resnet --attack=cw --iteration=100 --conf=30 --suffix_load="_2" --load_model=Q4 --test_batch_size=128
CUDA_VISIBLE_DEVICES=0 python trans.py  --dataset=cifar100 --save_path=./outputs/ --use_lib=advertorch --analysis=quant --arch=resnet --attack=cw --iteration=100 --conf=30 --suffix_load="_2" --load_model=Q6 --test_batch_size=128
CUDA_VISIBLE_DEVICES=0 python trans.py  --dataset=cifar100 --save_path=./outputs/ --use_lib=advertorch --analysis=quant --arch=resnet --attack=cw --iteration=100 --conf=30 --suffix_load="_2" --load_model=Q8 --test_batch_size=128
CUDA_VISIBLE_DEVICES=0 python trans.py  --dataset=cifar100 --save_path=./outputs/ --use_lib=advertorch --analysis=quant --arch=resnet --attack=cw --iteration=100 --conf=30 --suffix_load="_2" --load_model=HT --test_batch_size=128
CUDA_VISIBLE_DEVICES=0 python trans.py  --dataset=cifar100 --save_path=./outputs/ --use_lib=advertorch --analysis=quant --arch=resnet --attack=cw --iteration=100 --conf=30 --suffix_load="_3" --load_model=FP --test_batch_size=128
CUDA_VISIBLE_DEVICES=0 python trans.py  --dataset=cifar100 --save_path=./outputs/ --use_lib=advertorch --analysis=quant --arch=resnet --attack=cw --iteration=100 --conf=30 --suffix_load="_3" --load_model=Q1 --test_batch_size=128
CUDA_VISIBLE_DEVICES=0 python trans.py  --dataset=cifar100 --save_path=./outputs/ --use_lib=advertorch --analysis=quant --arch=resnet --attack=cw --iteration=100 --conf=30 --suffix_load="_3" --load_model=Q2 --test_batch_size=128
CUDA_VISIBLE_DEVICES=0 python trans.py  --dataset=cifar100 --save_path=./outputs/ --use_lib=advertorch --analysis=quant --arch=resnet --attack=cw --iteration=100 --conf=30 --suffix_load="_3" --load_model=Q4 --test_batch_size=128
CUDA_VISIBLE_DEVICES=0 python trans.py  --dataset=cifar100 --save_path=./outputs/ --use_lib=advertorch --analysis=quant --arch=resnet --attack=cw --iteration=100 --conf=30 --suffix_load="_3" --load_model=Q6 --test_batch_size=128
CUDA_VISIBLE_DEVICES=0 python trans.py  --dataset=cifar100 --save_path=./outputs/ --use_lib=advertorch --analysis=quant --arch=resnet --attack=cw --iteration=100 --conf=30 --suffix_load="_3" --load_model=Q8 --test_batch_size=128
CUDA_VISIBLE_DEVICES=0 python trans.py  --dataset=cifar100 --save_path=./outputs/ --use_lib=advertorch --analysis=quant --arch=resnet --attack=cw --iteration=100 --conf=30 --suffix_load="_3" --load_model=HT --test_batch_size=128
CUDA_VISIBLE_DEVICES=0 python trans.py  --dataset=cifar100 --save_path=./outputs/ --use_lib=advertorch --analysis=quant --arch=resnet --attack=cw --iteration=100 --conf=30 --suffix_load="_4" --load_model=FP --test_batch_size=128
CUDA_VISIBLE_DEVICES=0 python trans.py  --dataset=cifar100 --save_path=./outputs/ --use_lib=advertorch --analysis=quant --arch=resnet --attack=cw --iteration=100 --conf=30 --suffix_load="_4" --load_model=Q1 --test_batch_size=128
CUDA_VISIBLE_DEVICES=0 python trans.py  --dataset=cifar100 --save_path=./outputs/ --use_lib=advertorch --analysis=quant --arch=resnet --attack=cw --iteration=100 --conf=30 --suffix_load="_4" --load_model=Q2 --test_batch_size=128
CUDA_VISIBLE_DEVICES=0 python trans.py  --dataset=cifar100 --save_path=./outputs/ --use_lib=advertorch --analysis=quant --arch=resnet --attack=cw --iteration=100 --conf=30 --suffix_load="_4" --load_model=Q4 --test_batch_size=128
CUDA_VISIBLE_DEVICES=0 python trans.py  --dataset=cifar100 --save_path=./outputs/ --use_lib=advertorch --analysis=quant --arch=resnet --attack=cw --iteration=100 --conf=30 --suffix_load="_4" --load_model=Q6 --test_batch_size=128
CUDA_VISIBLE_DEVICES=0 python trans.py  --dataset=cifar100 --save_path=./outputs/ --use_lib=advertorch --analysis=quant --arch=resnet --attack=cw --iteration=100 --conf=30 --suffix_load="_4" --load_model=Q8 --test_batch_size=128
CUDA_VISIBLE_DEVICES=0 python trans.py  --dataset=cifar100 --save_path=./outputs/ --use_lib=advertorch --analysis=quant --arch=resnet --attack=cw --iteration=100 --conf=30 --suffix_load="_4" --load_model=HT --test_batch_size=128
CUDA_VISIBLE_DEVICES=0 python trans.py  --dataset=cifar100 --save_path=./outputs/ --use_lib=advertorch --analysis=quant --arch=resnet --attack=cw --iteration=100 --conf=30 --suffix_load="_5" --load_model=FP --test_batch_size=128
CUDA_VISIBLE_DEVICES=0 python trans.py  --dataset=cifar100 --save_path=./outputs/ --use_lib=advertorch --analysis=quant --arch=resnet --attack=cw --iteration=100 --conf=30 --suffix_load="_5" --load_model=Q1 --test_batch_size=128
CUDA_VISIBLE_DEVICES=0 python trans.py  --dataset=cifar100 --save_path=./outputs/ --use_lib=advertorch --analysis=quant --arch=resnet --attack=cw --iteration=100 --conf=30 --suffix_load="_5" --load_model=Q2 --test_batch_size=128
CUDA_VISIBLE_DEVICES=0 python trans.py  --dataset=cifar100 --save_path=./outputs/ --use_lib=advertorch --analysis=quant --arch=resnet --attack=cw --iteration=100 --conf=30 --suffix_load="_5" --load_model=Q4 --test_batch_size=128
CUDA_VISIBLE_DEVICES=0 python trans.py  --dataset=cifar100 --save_path=./outputs/ --use_lib=advertorch --analysis=quant --arch=resnet --attack=cw --iteration=100 --conf=30 --suffix_load="_5" --load_model=Q6 --test_batch_size=128
CUDA_VISIBLE_DEVICES=0 python trans.py  --dataset=cifar100 --save_path=./outputs/ --use_lib=advertorch --analysis=quant --arch=resnet --attack=cw --iteration=100 --conf=30 --suffix_load="_5" --load_model=Q8 --test_batch_size=128
CUDA_VISIBLE_DEVICES=0 python trans.py  --dataset=cifar100 --save_path=./outputs/ --use_lib=advertorch --analysis=quant --arch=resnet --attack=cw --iteration=100 --conf=30 --suffix_load="_5" --load_model=HT --test_batch_size=128


CUDA_VISIBLE_DEVICES=0 python trans.py  --dataset=cifar100 --save_path=./outputs/ --use_lib=advertorch --analysis=quant --arch=resnet --attack=PGD --suffix_load="_1" --load_model=FP --test_batch_size=128
CUDA_VISIBLE_DEVICES=0 python trans.py  --dataset=cifar100 --save_path=./outputs/ --use_lib=advertorch --analysis=quant --arch=resnet --attack=PGD --suffix_load="_1" --load_model=Q1 --test_batch_size=128
CUDA_VISIBLE_DEVICES=0 python trans.py  --dataset=cifar100 --save_path=./outputs/ --use_lib=advertorch --analysis=quant --arch=resnet --attack=PGD --suffix_load="_1" --load_model=Q2 --test_batch_size=128
CUDA_VISIBLE_DEVICES=0 python trans.py  --dataset=cifar100 --save_path=./outputs/ --use_lib=advertorch --analysis=quant --arch=resnet --attack=PGD --suffix_load="_1" --load_model=Q4 --test_batch_size=128
CUDA_VISIBLE_DEVICES=0 python trans.py  --dataset=cifar100 --save_path=./outputs/ --use_lib=advertorch --analysis=quant --arch=resnet --attack=PGD --suffix_load="_1" --load_model=Q6 --test_batch_size=128
CUDA_VISIBLE_DEVICES=0 python trans.py  --dataset=cifar100 --save_path=./outputs/ --use_lib=advertorch --analysis=quant --arch=resnet --attack=PGD --suffix_load="_1" --load_model=Q8 --test_batch_size=128
CUDA_VISIBLE_DEVICES=0 python trans.py  --dataset=cifar100 --save_path=./outputs/ --use_lib=advertorch --analysis=quant --arch=resnet --attack=PGD --suffix_load="_1" --load_model=HT --test_batch_size=128
# CUDA_VISIBLE_DEVICES=0 python trans.py  --dataset=cifar100 --save_path=./outputs/ --use_lib=advertorch --analysis=quant --arch=resnet --attack=PGD --suffix_load="_2" --load_model=FP --test_batch_size=128
# CUDA_VISIBLE_DEVICES=0 python trans.py  --dataset=cifar100 --save_path=./outputs/ --use_lib=advertorch --analysis=quant --arch=resnet --attack=PGD --suffix_load="_2" --load_model=Q1 --test_batch_size=128
# CUDA_VISIBLE_DEVICES=0 python trans.py  --dataset=cifar100 --save_path=./outputs/ --use_lib=advertorch --analysis=quant --arch=resnet --attack=PGD --suffix_load="_2" --load_model=Q2 --test_batch_size=128
# CUDA_VISIBLE_DEVICES=0 python trans.py  --dataset=cifar100 --save_path=./outputs/ --use_lib=advertorch --analysis=quant --arch=resnet --attack=PGD --suffix_load="_2" --load_model=Q4 --test_batch_size=128
# CUDA_VISIBLE_DEVICES=0 python trans.py  --dataset=cifar100 --save_path=./outputs/ --use_lib=advertorch --analysis=quant --arch=resnet --attack=PGD --suffix_load="_2" --load_model=Q6 --test_batch_size=128
# CUDA_VISIBLE_DEVICES=0 python trans.py  --dataset=cifar100 --save_path=./outputs/ --use_lib=advertorch --analysis=quant --arch=resnet --attack=PGD --suffix_load="_2" --load_model=Q8 --test_batch_size=128
# CUDA_VISIBLE_DEVICES=0 python trans.py  --dataset=cifar100 --save_path=./outputs/ --use_lib=advertorch --analysis=quant --arch=resnet --attack=PGD --suffix_load="_2" --load_model=HT --test_batch_size=128
# CUDA_VISIBLE_DEVICES=0 python trans.py  --dataset=cifar100 --save_path=./outputs/ --use_lib=advertorch --analysis=quant --arch=resnet --attack=PGD --suffix_load="_3" --load_model=FP --test_batch_size=128
# CUDA_VISIBLE_DEVICES=0 python trans.py  --dataset=cifar100 --save_path=./outputs/ --use_lib=advertorch --analysis=quant --arch=resnet --attack=PGD --suffix_load="_3" --load_model=Q1 --test_batch_size=128
# CUDA_VISIBLE_DEVICES=0 python trans.py  --dataset=cifar100 --save_path=./outputs/ --use_lib=advertorch --analysis=quant --arch=resnet --attack=PGD --suffix_load="_3" --load_model=Q2 --test_batch_size=128
# CUDA_VISIBLE_DEVICES=0 python trans.py  --dataset=cifar100 --save_path=./outputs/ --use_lib=advertorch --analysis=quant --arch=resnet --attack=PGD --suffix_load="_3" --load_model=Q4 --test_batch_size=128
# CUDA_VISIBLE_DEVICES=0 python trans.py  --dataset=cifar100 --save_path=./outputs/ --use_lib=advertorch --analysis=quant --arch=resnet --attack=PGD --suffix_load="_3" --load_model=Q6 --test_batch_size=128
# CUDA_VISIBLE_DEVICES=0 python trans.py  --dataset=cifar100 --save_path=./outputs/ --use_lib=advertorch --analysis=quant --arch=resnet --attack=PGD --suffix_load="_3" --load_model=Q8 --test_batch_size=128
# CUDA_VISIBLE_DEVICES=0 python trans.py  --dataset=cifar100 --save_path=./outputs/ --use_lib=advertorch --analysis=quant --arch=resnet --attack=PGD --suffix_load="_3" --load_model=HT --test_batch_size=128
# CUDA_VISIBLE_DEVICES=0 python trans.py  --dataset=cifar100 --save_path=./outputs/ --use_lib=advertorch --analysis=quant --arch=resnet --attack=PGD --suffix_load="_4" --load_model=FP --test_batch_size=128
# CUDA_VISIBLE_DEVICES=0 python trans.py  --dataset=cifar100 --save_path=./outputs/ --use_lib=advertorch --analysis=quant --arch=resnet --attack=PGD --suffix_load="_4" --load_model=Q1 --test_batch_size=128
# CUDA_VISIBLE_DEVICES=0 python trans.py  --dataset=cifar100 --save_path=./outputs/ --use_lib=advertorch --analysis=quant --arch=resnet --attack=PGD --suffix_load="_4" --load_model=Q2 --test_batch_size=128
# CUDA_VISIBLE_DEVICES=0 python trans.py  --dataset=cifar100 --save_path=./outputs/ --use_lib=advertorch --analysis=quant --arch=resnet --attack=PGD --suffix_load="_4" --load_model=Q4 --test_batch_size=128
# CUDA_VISIBLE_DEVICES=0 python trans.py  --dataset=cifar100 --save_path=./outputs/ --use_lib=advertorch --analysis=quant --arch=resnet --attack=PGD --suffix_load="_4" --load_model=Q6 --test_batch_size=128
# CUDA_VISIBLE_DEVICES=0 python trans.py  --dataset=cifar100 --save_path=./outputs/ --use_lib=advertorch --analysis=quant --arch=resnet --attack=PGD --suffix_load="_4" --load_model=Q8 --test_batch_size=128
# CUDA_VISIBLE_DEVICES=0 python trans.py  --dataset=cifar100 --save_path=./outputs/ --use_lib=advertorch --analysis=quant --arch=resnet --attack=PGD --suffix_load="_4" --load_model=HT --test_batch_size=128
# CUDA_VISIBLE_DEVICES=0 python trans.py  --dataset=cifar100 --save_path=./outputs/ --use_lib=advertorch --analysis=quant --arch=resnet --attack=PGD --suffix_load="_5" --load_model=FP --test_batch_size=128
# CUDA_VISIBLE_DEVICES=0 python trans.py  --dataset=cifar100 --save_path=./outputs/ --use_lib=advertorch --analysis=quant --arch=resnet --attack=PGD --suffix_load="_5" --load_model=Q1 --test_batch_size=128
# CUDA_VISIBLE_DEVICES=0 python trans.py  --dataset=cifar100 --save_path=./outputs/ --use_lib=advertorch --analysis=quant --arch=resnet --attack=PGD --suffix_load="_5" --load_model=Q2 --test_batch_size=128
# CUDA_VISIBLE_DEVICES=0 python trans.py  --dataset=cifar100 --save_path=./outputs/ --use_lib=advertorch --analysis=quant --arch=resnet --attack=PGD --suffix_load="_5" --load_model=Q4 --test_batch_size=128
# CUDA_VISIBLE_DEVICES=0 python trans.py  --dataset=cifar100 --save_path=./outputs/ --use_lib=advertorch --analysis=quant --arch=resnet --attack=PGD --suffix_load="_5" --load_model=Q6 --test_batch_size=128
# CUDA_VISIBLE_DEVICES=0 python trans.py  --dataset=cifar100 --save_path=./outputs/ --use_lib=advertorch --analysis=quant --arch=resnet --attack=PGD --suffix_load="_5" --load_model=Q8 --test_batch_size=128
# CUDA_VISIBLE_DEVICES=0 python trans.py  --dataset=cifar100 --save_path=./outputs/ --use_lib=advertorch --analysis=quant --arch=resnet --attack=PGD --suffix_load="_5" --load_model=HT --test_batch_size=128
