import matplotlib.pyplot as plt
import torch
from utils.load_dataset import load_dataset
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse
import eagerpy as ep
import foolbox
from foolbox.utils import accuracy, samples
from attack_framework.attacks import LinfPGDAttack_with_normalization
from utils.halftone import Halftone2d
from utils.quantise import Quantise2d
from models.resnet import *
from models.ensemble import Ensemble, Ensemble_Backpropable
import advertorch
from advertorch.utils import NormalizeByChannelMeanStd
from advertorch.bpda import BPDAWrapper
from utils.str2bool import str2bool
import os
import matplotlib.cm as cm


parser = argparse.ArgumentParser(description='Attack Ensemble', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--dataset',          default='CIFAR10', type=str,     help='Set dataset to use')
parser.add_argument('--parallel',         default=False,     type=bool,    help='Device in  parallel')
parser.add_argument('--device',           default=0,         type=int,     help='Device number')
parser.add_argument('--train_batch_size', default=32,        type=int,     help='Train batch size')
parser.add_argument('--test_batch_size',  default=32,        type=int,     help='Test batch size')
parser.add_argument('--attack',           default='PGD',     type=str,     help='Type of attack [PGD, CW]')
parser.add_argument('--visualize',        default=False,     type=bool,    help='Visualize adversarial images')
parser.add_argument('--use_bpda',         default=True,      type=bool,    help='Use Backward Pass through Differential Approximation when using attack')
parser.add_argument('--models',           default='FP,Q1,Q2',type=str,     help='Input Quantization for the model')
parser.add_argument('--suffixs',          default='_1,_1,_1',type=str,     help='Suffix for the models')
parser.add_argument('--iterations',       default=40,        type=int,     help='iterations for pgd attack')
parser.add_argument('--attack_type',      default='pgd',     type=str,     help='Grading selection for ensemble [avg, sign_avg, sign_all, transfer]')
parser.add_argument('--save_path',        default='./outputs/cifar10/transferability/',
                                                             type=str,     help='Save path for the output file')
global args
args = parser.parse_args()
print(args)
#--------------------------------------------------
# Parse input arguments
#--------------------------------------------------

# Parameters
batch_size    = args.train_batch_size
b_size_test   = args.test_batch_size
models        = args.models.split(',')
suffixs       = args.suffixs.split(',')

# Setup right device to run on
device = torch.device('cuda:'+str(args.device) if torch.cuda.is_available() else 'cpu')
train_loader, val_loader, test_loader, normalization_function, unnormalization_function, num_classes, mean, std,img_dim  = load_dataset(dataset=args.dataset, 
                                                                                                                                        train_batch_size=batch_size,
                                                                                                                                        test_batch_size=b_size_test, 
                                                                                                                                        device=device)
# Load model
print("Using Model: Ensemble of " + args.models)

if(args.attack_type == 'transfer'):
    model_surrogate_simple = Ensemble_Backpropable(device=device,
                                                   num_classes=num_classes,
                                                   dataset=args.dataset.lower(),
                                                   models=models)
    
    normalize = NormalizeByChannelMeanStd(mean=mean, std=std)
    ensemble_surrogate = nn.Sequential(normalize, model_surrogate_simple).to(device)

    # Super important not to forget this statement, cost 2 development days to figure out why
    # numbers were so wrong with ensemble_surrogate
    ensemble_surrogate.eval()

    model = Ensemble(device=device,
                     dataset=args.dataset.lower(),
                     num_classes=num_classes,
                     quant=models,
                     suffix=suffixs)
else:
    model = Ensemble(device=device,
                     dataset=args.dataset.lower(),
                     num_classes=num_classes,
                     quant=models,
                     suffix=suffixs)


if args.parallel:
    model = nn.DataParallel(model, device_ids=[0,1,2,3])
else:
    model = model.to(device)
    print('Warning: Not using DataParallel')

model.eval()

# Instantiating custom normalization function and dataloading
global attack

EPSILON = 8
iterations = args.iterations
stepsize = 0.01
epsilons = [0.01, 0.02, 0.03, 0.04, 0.06, 0.08, 0.1]
attack = LinfPGDAttack_with_normalization(clip_min=0.0, 
                                          clip_max=1.0,
                                          epsilon=EPSILON*1.0/255.0,
                                          k=iterations,
                                          a=stepsize,
                                          random_start=True,
                                          loss_func='xent')

if(args.use_bpda):
    print("Attack method: " + args.attack + " with BPDA")
else:
    print("Attack method: " + args.attack + " without BPDA")

def test(model, device, test_loader, epsilon):
    # Accuracy counter
    correct = 0
    total = 0
    L2 = 0
    Linf = 0
    global attack

    # Loop over all examples in test set
    for _, (data, target) in enumerate(test_loader):
        attack.epsilon = epsilon

        # Send the data and label to the device
        data, target = data.to(device), target.to(device)
        grad_list = None
        if(args.use_bpda):
            for quant, net in zip(model.quantization, model.models):
                un_norm_perturbed_data = attack.perturb_bpda(data,
                                                             target, 
                                                             net, 
                                                             normalization_function,
                                                             forward=quant.forward,
                                                             backward_replacement=quant.back_approx)

                grad = attack.get_grad(un_norm_perturbed_data, target, net, normalization_function, quant.forward, quant.back_approx)
                grad = grad.detach().clone()

                if(grad_list == None):
                    grad_list = torch.stack([grad], dim=0)
                else:
                    grad_list = torch.cat([grad_list, grad.unsqueeze(0)], dim=0)

        return grad_list.cpu().numpy(), data.cpu().numpy()

accuracies = []
examples = []

fig, ax = plt.subplots(nrows=4, 
                       ncols=5,
                       figsize=(8, 8))

def get_grad_sign(grad, model, batch_index):
    return np.sign(grad[model][batch_index]).transpose(1, 2, 0)

def get_normalized_grad_img(grad, model, batch_index, ret_sum=False):
    img = grad[model][batch_index]

    sum_img = (grad[0][batch_index] + grad[1][batch_index] + grad[2][batch_index]) / 3


    min_val = grad[:, batch_index, :].min()
    max_val = grad[:, batch_index, :].max()

    sum_img_max = sum_img.max()
    sum_img_min = sum_img.min()

    max_val = max(max_val, sum_img_max)
    min_val = min(min_val, sum_img_min)

    range_val = max_val - min_val

    img = (img - min_val) / range_val

    if(ret_sum):
        sum_img = (sum_img - min_val) / range_val
        return sum_img.transpose(1,2,0)
    return img.transpose(1,2,0)

for row in range(4):
    for col in range(5):
        ax[row][col].set_axis_off()

# Run test for each epsilon
torch.cuda.empty_cache()
grad, data = test(model, device, test_loader, 0.01)
for i in range(args.test_batch_size):
    img = data[i].transpose(1,2,0)
    ax[0][0].imshow(img)
    ax[0][2].imshow(get_normalized_grad_img(grad, 0, i), cmap=cm.gist_rainbow)
    ax[0][3].imshow(get_normalized_grad_img(grad, 1, i), cmap=cm.gist_rainbow)
    ax[0][4].imshow(get_normalized_grad_img(grad, 2, i), cmap=cm.gist_rainbow)
    ax[3][0].imshow(get_normalized_grad_img(grad, 2, i, ret_sum=True), cmap=cm.gist_rainbow)
    s1 = get_grad_sign(grad, 0, i)
    s2 = get_grad_sign(grad, 1, i)
    s3 = get_grad_sign(grad, 2, i)
    s_avg = s1 + s2 + s3
    avg = (grad[0][i] + grad[1][i] + grad[2][i]) / 3
    sg_avg = np.sign(avg).transpose(1, 2, 0)
    ax[1][2].imshow(s1)
    ax[1][2].title.set_text("FP Grad Dir.\n(S1)")
    ax[1][3].imshow(s2)
    ax[1][3].title.set_text("Q1 Grad Dir.\n(S2)")
    ax[1][4].imshow(s3)
    ax[1][4].title.set_text("Q2 Grad Dir.\n(S3)")
    ax[2][1].imshow(s_avg)
    ax[2][1].title.set_text("A-GD")
    ax[2][2].imshow(s_avg - s1)
    ax[2][2].title.set_text("A-GD - S1")
    ax[2][3].imshow(s_avg - s2)
    ax[2][3].title.set_text("A-GD - S2")
    ax[2][4].imshow(s_avg - s3)
    ax[2][4].title.set_text("A-GD - S3")
    ax[3][1].imshow(sg_avg)
    ax[3][1].title.set_text("Dir. of Avg. Grad\n (D-AG)")
    ax[3][2].imshow(sg_avg - s1)
    ax[3][2].title.set_text("D-AG - S1")
    ax[3][3].imshow(sg_avg - s2)
    ax[3][3].title.set_text("D-AG - S2")
    ax[3][4].imshow(sg_avg - s3)
    ax[3][4].title.set_text("D-AG - S3")
    ax[3][4].set_axis_off()

    ax[0][0].title.set_text('Image')
    ax[0][2].title.set_text('FP PGD Grad.')
    ax[0][3].title.set_text('Q1 PGD Grad.')
    ax[0][4].title.set_text('Q2 PGD Grad.')
    ax[3][0].title.set_text('Avg. Grad.')
    plt.subplots_adjust(wspace=0.5)
    plt.show()
    plt.savefig("./outputs/out{}.eps".format(i))
    plt.cla()


