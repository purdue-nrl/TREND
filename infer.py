import torch
import torchvision
import torch.nn as nn
from torchvision.transforms import transforms
from utils.instantiate_model import instantiate_model
from utils.load_dataset import load_dataset
import argparse
import os, sys
import torchvision.models as models
from utils.str2bool import str2bool
from utils.inference import inference
from utils.load_dataset import load_dataset
from attack_framework.multi_lib_attacks import attack_wrapper

parser = argparse.ArgumentParser(description='Train', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# Training parameters
parser.add_argument('--dataset',                default='CIFAR10',      type=str,       help='Set dataset to use')
parser.add_argument('--batch_size',                default=512,      type=int,       help="batch size")
parser.add_argument('--loss',                   default='crossentropy', type=str,       help='loss function for training')

# Dataloader args
parser.add_argument('--test_batch_size',        default=512,            type=int,       help='Test batch size')

# Model parameters
parser.add_argument('--load_model',             default='FP',           type=str,       help='Quantization transfer function-Q1 Q2 Q4 Q6 Q8 HT FP')
parser.add_argument('--suffix',                 default='',             type=str,       help='appended to model name')
parser.add_argument('--dorefa',                 default=False,          type=str2bool,  help='Use Dorefa Net')
parser.add_argument('--arch',                   default='resnet',       type=str,       help='Network architecture')
parser.add_argument('--qout',                   default=False,          type=str2bool,  help='Output layer weight quantisation')
parser.add_argument('--qin',                    default=False,          type=str2bool,  help='Input layer weight quantisation')
parser.add_argument('--abit',                   default=32,             type=int,       help='Activation quantisation precision')
parser.add_argument('--wbit',                   default=32,             type=int,       help='Weight quantisation precision')
global args
args = parser.parse_args()
print(args)

suffixs = ["_1", "_2", "_3", "_4", "_5"]
# Setup right device to run on
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("\n")
train_loader, _,test_loader, normalization_function, unnormalization_function,num_classes, mean, std, img_dim = load_dataset(dataset=args.dataset, 
                                                                                                                            train_batch_size = args.batch_size,
                                                                                                                            test_batch_size=args.batch_size, 
                                                                                                                            val_split=0, 
                                                                                                                            device=device)

model_names = []
networks={}


for suf in suffixs:
    #Instantiate model 
    net, model_name, Q = instantiate_model(dataset=args.dataset,
                                        num_classes=num_classes,
                                        load_model = args.load_model,
                                        q_tf = args.load_model,
                                        arch=args.arch,
                                        dorefa=args.dorefa,
                                        abit=args.abit, 
                                        wbit=args.wbit,
                                        qin=args.qin,
                                        qout=args.qout,
                                        suffix=args.suffix+suf, 
                                        device=device, 
                                        load = True)
    model_names.append(model_name)
    networks[model_name] = (Q, net)


test_acc = []
train_acc = []
for model_name in model_names:
    test_correct, test_total, test_accuracy = inference(Q=networks[model_name][0], normalization_function = normalization_function, net=networks[model_name][1], data_loader = test_loader, device=device)
    print(' Test set: Accuracy: {}/{} ({:.2f}%)'.format(
                test_correct, test_total, test_accuracy))
    test_acc.append(test_accuracy)
    # train_correct, train_total, train_accuracy = inference(Q=networks[model_name][0], normalization_function = normalization_function, net=networks[model_name][1], data_loader = train_loader, device=device)
    # print(' Train set: Accuracy: {}/{} ({:.2f}%)'.format(
    #             train_correct, train_total,train_accuracy) )
    # train_acc.append(train_accuracy)
#import Math
test_acc = torch.tensor(test_acc)
mean = torch.mean(test_acc)
std = torch.std(test_acc)
print(mean, std)