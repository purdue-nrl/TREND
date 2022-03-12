import numpy as np
from matplotlib import pyplot as plt 
import torch
import argparse
from scipy import optimize
from scipy.stats.distributions import  t
import sys
from utils.str2bool import str2bool
import math
import os 
parser = argparse.ArgumentParser(description='Predict Ensemble Accuracy', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--models',           default='FP_resnet',type=str,     help='Model 2 for ensemble, set to [Q1, Q2, HT, A1, Q1A1, Q2A1, HTA1]')
parser.add_argument('--dataset',          default='cifar10',type=str,     help='dataset cifar10 imagenet simple basic mnist')
parser.add_argument('--attack_type',      default='avg',type=str,     help='Grading selection for ensemble [avg, sign_avg, sign_all, transfer]')
parser.add_argument('--save_path',        default='../outputs/cifar10/transferability/',
                                                             type=str,     help='Save path for the output file')
parser.add_argument('--dorefa_test',           default=False,type=str2bool,     help='loads act/wt quant models')
parser.add_argument('--accuracy',   default=1.0, type=float, help="Accuracy of the ensemble")
parser.add_argument('--subsample', default=3, type=int, help="Subsample the training data")

global args
args = parser.parse_args()
print(args)
models  = args.models.split(',')
model_list = [('FP_resnet'),
              ('FP_resnet_2'),
              ('FP_resnet_3'),
              ('FP_resnet_4')]

file_name = args.dataset
transfered_images = []
#load data in required format
for i in range(len(models)):
    file_name += '_'+ models[i]
    current_model_data = torch.load(args.save_path + args.dataset +'_'+ models[i] +'.pt')
    current_model_transfered_images = torch.cat(current_model_data['num_trans_images'], dim=1).unsqueeze(dim=1)
    transfered_images.append(current_model_transfered_images)

# Tensor Target models, Source model, Epsilon
transfered_images = torch.cat (transfered_images, dim=1).cpu().numpy()
epsilon_transfer = current_model_data['epsilons']

# Compute the fitting with the data
exp_coef = np.zeros((transfered_images.shape[0], transfered_images.shape[0], 2))
exp_cov = np.zeros((transfered_images.shape[0], transfered_images.shape[0], 2,2))
alpha=0.05
dof=transfered_images.shape[2]-exp_cov.shape[2]
tval = t.ppf(1.0-alpha/2.0,dof)

# Get the coefficients of the fit
for i in range(transfered_images.shape[0]):
    for j in range(transfered_images.shape[1]):
        exp_coef[i,j],exp_cov[i,j] = optimize.curve_fit(lambda t,a,b: a*(1-np.exp(b*t)),  epsilon_transfer, transfered_images[i, j], p0=(10000, -150), maxfev=10000)

epsilons = np.array([8/255.0])
TM = np.zeros((len(epsilons), transfered_images.shape[0]*(transfered_images.shape[1]) ))

for i in range(transfered_images.shape[0]):
    for  j in range(transfered_images.shape[1]):
        total_SE = np.sum((exp_coef[i,j,0]*(1-np.exp(exp_coef[i,j,1] * epsilon_transfer)) - transfered_images[i,j])**2)
        Train_SE = np.sum( (exp_coef[i,j,0]*(1-np.exp(exp_coef[i,j,1] * epsilon_transfer))[::int(transfered_images.shape[2]/args.subsample)] - transfered_images[i,j,::int(transfered_images.shape[2]/args.subsample)])**2 ) 
        Test_SE =  total_SE-Train_SE
        Train_RMSE =  np.sqrt(Train_SE /(transfered_images[i,j,::int(transfered_images.shape[2]/args.subsample ) ].shape[0]) )  
        Test_RMSE = np.sqrt(Test_SE / (transfered_images.shape[2]-transfered_images[i,j,::int(transfered_images.shape[2]/args.subsample)].shape[0]))
        
        plt.figure(figsize=(20, 15))
        plt.plot(epsilon_transfer, exp_coef[i,j,0]*(1-np.exp(exp_coef[i,j,1] * epsilon_transfer)),label='predicted')
        plt.plot(epsilon_transfer, transfered_images[i,j],label='actual')
        plt.grid()
        plt.legend()
        plt.ylabel("Number of images")
        plt.xlabel("Epsilon")
        plt.title(" TRAIN:{} TEST:{} #_Of_Points:{}".format(Train_RMSE,Test_RMSE, transfered_images[0,0,::int(transfered_images.shape[2]/args.subsample)].shape[0] ) )
        plt.show()
        plt.close()


