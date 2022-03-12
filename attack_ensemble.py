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


parser = argparse.ArgumentParser(description='Attack Ensemble', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--dataset',          default='CIFAR10',     type=str,     help='Set dataset to use')
parser.add_argument('--parallel',         default=False,         type=bool,    help='Device in  parallel')
parser.add_argument('--device',           default=0,             type=int,     help='Device number')
parser.add_argument('--train_batch_size', default=256,            type=int,     help='Train batch size')
parser.add_argument('--test_batch_size',  default=256,            type=int,     help='Test batch size')
parser.add_argument('--attack',           default='PGD',         type=str,     help='Type of attack [PGD, CW]')
parser.add_argument('--visualize',        default=False,         type=bool,    help='Visualize adversarial images')
parser.add_argument('--use_bpda',         default=True,          type=bool,    help='Use Backward Pass through Differential Approximation when using attack')
parser.add_argument('--models',           default='FP,FP,FP',    type=str,     help='Input Quantization for the model')
parser.add_argument('--suffixs',          default='_1,_2,_3',    type=str,     help='Suffix for the models')
parser.add_argument('--iterations',       default=40,            type=int,     help='iterations for pgd attack')
parser.add_argument('--attack_type',      default='sign_avg',    type=str,     help='Grading selection for ensemble [avg, sign_avg, sign_all, transfer]')
parser.add_argument('--save_path',        default='./outputs/',  type=str,     help='Save path for the output file')
parser.add_argument('--arch',             default='resnet',      type=str,     help='Network architecture')
global args
args = parser.parse_args()
print(args)
#--------------------------------------------------
# Parse input arguments
# Note sign_avg is the same as ASG attack
#--------------------------------------------------

# Parameters
batch_size     = args.train_batch_size
b_size_test    = args.test_batch_size
models         = args.models.split(',')
suffixs        = args.suffixs.split(',')
args.arch      = args.arch.split(',')
args.save_path = os.path.join(args.save_path, args.dataset.lower(), 'ensemble_attacks/')

# Setup right device to run on
device = torch.device('cuda:'+ str(args.device) if torch.cuda.is_available() else 'cpu')
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
                     arch=args.arch,
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
epsilons = [0.01]#[0.031]#[0.01, 0.015, 0.02, 0.03, 0.04, 0.06, 0.08, 0.1]
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
    for batch_idx, (data, target) in enumerate(test_loader):
        attack.epsilon = epsilon

        # Send the data and label to the device
        data, target = data.to(device), target.to(device)
        if(args.attack_type =='transfer'):
            # Implementing attack model described in https://arxiv.org/pdf/2002.08347.pdf section 12.3
            # On Adaptive Attacks to Adversarial Example Defenses by Florian Tramer, Nicholas Carlini, Wieland Brendel, Aleksander Madry            
            adversary = advertorch.attacks.LinfPGDAttack(ensemble_surrogate, eps=epsilon, eps_iter=stepsize, nb_iter=iterations, rand_init=True, targeted=False)
            un_norm_perturbed_data = adversary.perturb(data, target)
            perturbed_data = normalize(un_norm_perturbed_data)
        else:
            if(args.use_bpda):
                un_norm_perturbed_data = attack.perturb_bpda_ensemble(data, target, model, normalization_function, attack_type=args.attack_type)
                perturbed_data = normalization_function(un_norm_perturbed_data) #<<== make sure to normalize before fetching to the model
            else:
                un_norm_perturbed_data = attack.perturb_ensemble(data, target, model, normalization_function) #<<== pass normalization function to attack
                perturbed_data = normalization_function(un_norm_perturbed_data) #<<== make sure to normalize before fetching to the model 
      
        with torch.no_grad():
            # Re-classify the perturbed image
            output = model(perturbed_data)
            
            if(args.visualize):
                for i in range(10):
                    plt.subplot(121)
                    plt.imshow(data[i].detach().cpu().numpy().transpose(1,2,0))

                    plt.subplot(122)
                    plt.imshow(perturbed_data[i].detach().cpu().numpy().transpose(1,2,0))

            # Check for success
            final_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            final_pred = final_pred.squeeze(dim=1)

            correct += (final_pred == target).sum()
            total += data.shape[0]

        L2 += torch.sum(torch.norm(data - un_norm_perturbed_data, p=2, dim=(1,2,3)))
        Linf += torch.sum(torch.norm(data - un_norm_perturbed_data, p=float('inf'), dim=(1,2,3)))

        norm_2 = float(L2.item())/ float(total)
        norm_inf = float(Linf.item())/ float(total)

    # Calculate final accuracy for this epsilon
    final_acc = float(correct) * 100 / float(total)
    norm_2 = float(L2.item())/float(total)
    norm_inf = float(Linf.item())/float(total)

    # Return the accuracy and an adversarial example
    return final_acc

accuracies = []
examples = []
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        labels = labels.to(device)
        images = images.to(device)
        images = normalization_function(images)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

baseline = correct * 100.0 / total
print("Baseline clean accuracy {:.4f}".format(baseline))

# Run test for each epsilon
for eps in epsilons:
    torch.cuda.empty_cache()
    acc = test(model, device, test_loader, eps)
    accuracies.append(acc)

print("Final accuracies:")
print(accuracies)
file_name = args.dataset.lower()

for i in range(len(models)):
    file_name += '_'+ models[i]

data = {'dataset': args.dataset.lower(),
        'baseline': baseline,
        'accuracy': accuracies,
        'epsilons': epsilons}

torch.save(data, os.path.join(args.save_path , file_name + "_" + args.attack_type +'.pt'))