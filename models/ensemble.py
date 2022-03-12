import os
import torch
import torch.nn as nn
import numpy as np
from models.resnet import *
from models.vgg import *
import torchvision.models as models
from utils.instantiate_model import instantiate_model
import advertorch
from advertorch.bpda import BPDAWrapper
import copy
import torchvision.models as torchmodels


class Ensemble(nn.Module):
    def __init__(self,
                 device,
                 num_classes=10,
                 arch=['resnet', 'resenet', 'resnet'],
                 quant=['FP','Q1','Q2'],
                 suffix=['_1','_2','_3'],
                 abit=32,
                 wbit=32,
                 dorefa=False,
                 dataset='cifar10',
                 pretrn=False):
        super(Ensemble, self).__init__()

        # Holds all the ensemble model types
        self.quant = quant
        self.device = device
        self.dataset = dataset
        self.suffix = suffix
        self.abit = abit
        self.wbit = wbit
        self.dorefa = dorefa
        self.num_classes = num_classes
        self.net_type = 'ensemble'
        self.quantization = []
        self.models = []

        print("Loading Ensemble")
        for quant, suff, model_arch in zip(self.quant, suffix, arch):
            model = self.get_model(dataset, quant, model_arch, suff)
            model.eval()
            self.models.append(model)

        self.num_models = len(self.models)

    def get_model(self, dataset, quant, arch, suffix):
        # See prototype design pattern https://en.wikipedia.org/wiki/Prototype_pattern
        # This dictionary holds the prototypes

        net, model_name, Q = instantiate_model(dataset=self.dataset,
                                               num_classes=self.num_classes,
                                               load_model=quant,
                                               q_tf=quant,
                                               arch=arch,
                                               dorefa=self.dorefa,
                                               abit=self.abit, 
                                               wbit=self.wbit,
                                               qin=False,
                                               qout=False,
                                               suffix=suffix, 
                                               device=self.device,
                                               load=True)
        self.quantization.append(Q)
        return net

    def forward(self, inputs):
        model_out = []
        for model, quant in zip(self.models, self.quantization):
            model_output = model(quant(inputs))
            model_out.append(model_output)

        predMax = []
        # A simple voting mechanism
        # Find the max value of logits
        # Find the index of that location and convert to one hot
        # Add the one hot values to get the final output
        for output in model_out:
            predMax.append(torch.argmax(output, dim=1))
 
        out = torch.zeros([inputs.shape[0], self.num_classes]).to(self.device)
        for pred in predMax:
            model_vote = nn.functional.one_hot(pred, num_classes=self.num_classes).float()
            out += model_vote

        # If none of the networks agree pick a random's networks prediction as output
        disagree_indices = torch.where(out.max(dim=1)[0] == 1)[0]

        # Choose a network at random whose output is chosen when none of them agree
        choose_networks = torch.LongTensor(len(disagree_indices)).random_(0, self.num_models)

        # Stack all the outputs, so we can choose randomly from among them
        pred = torch.stack(predMax, dim=0)

        # Set output from the networks whose output we choose randomly 
        out[disagree_indices] = nn.functional.one_hot(pred[choose_networks, disagree_indices], num_classes=self.num_classes).float()
        return out

class Ensemble_Backpropable(nn.Module):
    def __init__(self,
                 device,
                 num_classes=10,
                 models=['FP','Q1','Q2','HT'],
                 dataset='cifar10',
                 pretrn=False):
        super(Ensemble_Backpropable, self).__init__()

        # Holds all the ensemble model types
        self.model_types = models
        self.device = device
        self.num_classes = num_classes
        self.net_type = 'ensemble'

        self.model_path_dict = {
            'FP' : os.path.join('./pretrained', dataset, dataset + '_FP_' + 'resnet.ckpt'),
            'Q1' : os.path.join('./pretrained', dataset, dataset + '_Q1_' + 'resnet.ckpt'),
            'Q2' : os.path.join('./pretrained', dataset, dataset + '_Q2_' + 'resnet.ckpt'),
            'Q4' : os.path.join('./pretrained', dataset, dataset + '_Q4_' + 'resnet.ckpt'),
            'Q6' : os.path.join('./pretrained', dataset, dataset + '_Q6_' + 'resnet.ckpt'),
            'Q8' : os.path.join('./pretrained', dataset, dataset + '_Q8_' + 'resnet.ckpt'),
            'HT' : os.path.join('./pretrained', dataset, dataset + '_HT_' + 'resnet.ckpt'),
            'Q1_resnet' : os.path.join('./pretrained', dataset, dataset + '_Q1_' + 'resnet.ckpt'),
            'Q2_resnet' : os.path.join('./pretrained', dataset, dataset + '_Q2_' + 'resnet.ckpt'),
            'Q4_resnet' : os.path.join('./pretrained', dataset, dataset + '_Q4_' + 'resnet.ckpt'),
            'Q6_resnet' : os.path.join('./pretrained', dataset, dataset + '_Q6_' + 'resnet.ckpt'),
            'Q8_resnet' : os.path.join('./pretrained', dataset, dataset + '_Q8_' + 'resnet.ckpt'),
            'HT_resnet' : os.path.join('./pretrained', dataset, dataset + '_HT_' + 'resnet.ckpt'),
            'A1' : os.path.join('./pretrained', dataset, dataset + '_FP_' + 'resnet_a1w32.ckpt'),
            'Q1A1' : os.path.join('./pretrained', dataset, dataset + '_Q1_' + 'resnet_a1w32.ckpt'),
            'Q2A1' : os.path.join('./pretrained', dataset, dataset + '_Q2_' + 'resnet_a1w32.ckpt'),
            'HTA1' : os.path.join('./pretrained', dataset, dataset + '_HT_' + 'resnet_a1w32.ckpt'),
            'A2W2' : os.path.join('./pretrained', dataset, dataset + '_FP_' + 'resnet_a2w2.ckpt'),
            'A4W2' : os.path.join('./pretrained', dataset, dataset + '_FP_' + 'resnet_a4w2.ckpt'),

            'FP_vgg11' : os.path.join('./pretrained', dataset, dataset + '_FP_' + 'vgg11.ckpt'),
            'FP_vgg11_2' : os.path.join('./pretrained', dataset, dataset + '_FP_' + 'vgg11_2.ckpt'),
            'FP_vgg11_3' : os.path.join('./pretrained', dataset, dataset + '_FP_' + 'vgg11_3.ckpt'),
            'FP_vgg11_4' : os.path.join('./pretrained', dataset, dataset + '_FP_' + 'vgg11_4.ckpt'),
            'FP_vgg11bn' : os.path.join('./pretrained', dataset, dataset + '_FP_' + 'vgg11bn.ckpt'),
            'FP_vgg11bn_2' : os.path.join('./pretrained', dataset, dataset + '_FP_' + 'vgg11bn_2.ckpt'),
            'FP_vgg11bn_3' : os.path.join('./pretrained', dataset, dataset + '_FP_' + 'vgg11bn_3.ckpt'),
            'FP_vgg11bn_4' : os.path.join('./pretrained', dataset, dataset + '_FP_' + 'vgg11bn_4.ckpt'),
            'FP_resnet' : os.path.join('./pretrained', dataset, dataset + '_FP_' + 'resnet.ckpt'),
            'FP_resnet_2' : os.path.join('./pretrained', dataset, dataset + '_FP_' + 'resnet_2.ckpt'),
            'FP_resnet_3' : os.path.join('./pretrained', dataset, dataset + '_FP_' + 'resnet_3.ckpt'),
            'FP_resnet_4' : os.path.join('./pretrained', dataset, dataset + '_FP_' + 'resnet_4.ckpt'),

            'FP_torch_resnet18' : os.path.join('./pretrained', dataset, dataset + '_FP' + '_torch_resnet18.ckpt'),
            'Q1_torch_resnet18' : os.path.join('./pretrained', dataset, dataset + '_Q1' + '_torch_resnet18.ckpt'),
            'Q2_torch_resnet18' : os.path.join('./pretrained', dataset, dataset + '_Q2' + '_torch_resnet18.ckpt'),
            'Q4_torch_resnet18' : os.path.join('./pretrained', dataset, dataset + '_Q4' + '_torch_resnet18.ckpt'),
            'Q6_torch_resnet18' : os.path.join('./pretrained', dataset, dataset + '_Q6' + '_torch_resnet18.ckpt'),
            'Q8_torch_resnet18' : os.path.join('./pretrained', dataset, dataset + '_Q8' + '_torch_resnet18.ckpt'),
            'HT_torch_resnet18' : os.path.join('./pretrained', dataset, dataset + '_HT' + '_torch_resnet18.ckpt'),
            
            'FP_torch_vgg11' : os.path.join('./pretrained', dataset, dataset + '_FP_' + 'torch_vgg11.ckpt'),
            'FP_torch_vgg11_2' : os.path.join('./pretrained', dataset, dataset + '_FP_' + 'torch_vgg11_2.ckpt'),
            'FP_torch_vgg11_3' : os.path.join('./pretrained', dataset, dataset + '_FP_' + 'torch_vgg11_3.ckpt'),
            'FP_torch_vgg11_4' : os.path.join('./pretrained', dataset, dataset + '_FP_' + 'torch_vgg11_4.ckpt'),
            'FP_torch_vgg13' : os.path.join('./pretrained', dataset, dataset + '_FP_' + 'torch_vgg13.ckpt'),
            'FP_torch_vgg16' : os.path.join('./pretrained', dataset, dataset + '_FP_' + 'torch_vgg16.ckpt'),
            'FP_torch_vgg19' : os.path.join('./pretrained', dataset, dataset + '_FP_' + 'torch_vgg19.ckpt'),
            'FP_torch_vgg11bn' : os.path.join('./pretrained', dataset, dataset + '_FP_' + 'torch_vgg11bn.ckpt'),
            'FP_torch_vgg11bn_2' : os.path.join('./pretrained', dataset, dataset + '_FP_' + 'torch_vgg11bn_2.ckpt'),
            'FP_torch_vgg11bn_3' : os.path.join('./pretrained', dataset, dataset + '_FP_' + 'torch_vgg11bn_3.ckpt'),
            'FP_torch_vgg11bn_4' : os.path.join('./pretrained', dataset, dataset + '_FP_' + 'torch_vgg11bn_4.ckpt'),
            'FP_torch_vgg13bn' : os.path.join('./pretrained', dataset, dataset + '_FP_' + 'torch_vgg13bn.ckpt'),
            'FP_torch_vgg16bn' : os.path.join('./pretrained', dataset, dataset + '_FP_' + 'torch_vgg16bn.ckpt'),
            'FP_torch_vgg19bn' : os.path.join('./pretrained', dataset, dataset + '_FP_' + 'torch_vgg19bn.ckpt'),
            'FP_torch_resnet18_2' : os.path.join('./pretrained', dataset, dataset + '_FP_' + 'torch_resnet_2.ckpt'),
            'FP_torch_resnet18_3' : os.path.join('./pretrained', dataset, dataset + '_FP_' + 'torch_resnet_3.ckpt'),
            'FP_torch_resnet18_4' : os.path.join('./pretrained', dataset, dataset + '_FP_' + 'torch_resnet_4.ckpt'),

        }

        # See prototype design pattern https://en.wikipedia.org/wiki/Prototype_pattern
        # This dictionary holds the prototypes
        self.model_prototype_dict = {
            'FP' : ResNet18(num_classes=num_classes),
            'Q1' : ResNet18(num_classes=num_classes),
            'Q2' : ResNet18(num_classes=num_classes),
            'Q4' : ResNet18(num_classes=num_classes),
            'Q6' : ResNet18(num_classes=num_classes),
            'Q8' : ResNet18(num_classes=num_classes),
            'HT' : ResNet18(num_classes=num_classes),
            'Q1_resnet' : ResNet18(num_classes=num_classes),
            'Q2_resnet' : ResNet18(num_classes=num_classes),
            'Q4_resnet' : ResNet18(num_classes=num_classes),
            'Q6_resnet' : ResNet18(num_classes=num_classes),
            'Q8_resnet' : ResNet18(num_classes=num_classes),
            'HT_resnet' : ResNet18(num_classes=num_classes),
            'A1' : ResNet18_Dorefa(num_classes=num_classes, abit=1, wbit=32),
            'Q1A1' : ResNet18_Dorefa(num_classes=num_classes, abit=1, wbit=32),
            'Q2A1' : ResNet18_Dorefa(num_classes=num_classes, abit=1, wbit=32),
            'HTA1' : ResNet18_Dorefa(num_classes=num_classes, abit=1, wbit=32),
            'A2W2' : ResNet18_Dorefa(num_classes=num_classes, abit=2, wbit=2),
            'A4W2' : ResNet18_Dorefa(num_classes=num_classes, abit=4, wbit=2),

            'FP_vgg11' : vgg(cfg='11', batch_norm_conv=False, batch_norm_linear=False ,num_classes=num_classes),
            'FP_vgg11_2' : vgg(cfg='11', batch_norm_conv=False, batch_norm_linear=False ,num_classes=num_classes),
            'FP_vgg11_3' : vgg(cfg='11', batch_norm_conv=False, batch_norm_linear=False ,num_classes=num_classes),
            'FP_vgg11_4' : vgg(cfg='11', batch_norm_conv=False, batch_norm_linear=False ,num_classes=num_classes),
            'FP_vgg11bn' : vgg(cfg='11', batch_norm_conv=True, batch_norm_linear=False ,num_classes=num_classes),
            'FP_vgg11bn_2' : vgg(cfg='11', batch_norm_conv=True, batch_norm_linear=False ,num_classes=num_classes),
            'FP_vgg11bn_3' : vgg(cfg='11', batch_norm_conv=True, batch_norm_linear=False ,num_classes=num_classes),
            'FP_vgg11bn_4' : vgg(cfg='11', batch_norm_conv=True, batch_norm_linear=False ,num_classes=num_classes),
            'FP_resnet' : ResNet18(num_classes=num_classes),
            'FP_resnet_2' : ResNet18(num_classes=num_classes),
            'FP_resnet_3' : ResNet18(num_classes=num_classes),
            'FP_resnet_4' : ResNet18(num_classes=num_classes),

            'FP_torch_resnet18' : torchmodels.resnet18(pretrained=pretrn),
            'Q1_torch_resnet18' : torchmodels.resnet18(pretrained=pretrn),
            'Q2_torch_resnet18' : torchmodels.resnet18(pretrained=pretrn),
            'Q6_torch_resnet18' : torchmodels.resnet18(pretrained=pretrn),
            'Q8_torch_resnet18' : torchmodels.resnet18(pretrained=pretrn),
            'HT_torch_resnet18' : torchmodels.resnet18(pretrained=pretrn),
            
            'FP_torch_vgg11' : torchmodels.vgg11(pretrained=pretrn),
            'FP_torch_vgg11_2' : torchmodels.vgg11(pretrained=pretrn),
            'FP_torch_vgg11_3' : torchmodels.vgg11(pretrained=pretrn),
            'FP_torch_vgg11_4' : torchmodels.vgg11(pretrained=pretrn),
            'FP_torch_vgg13' : torchmodels.vgg13(pretrained=pretrn),
            'FP_torch_vgg16' : torchmodels.vgg16(pretrained=pretrn),
            'FP_torch_vgg19' : torchmodels.vgg19(pretrained=pretrn),
            'FP_torch_vgg11bn' : torchmodels.vgg11_bn(pretrained=pretrn),
            'FP_torch_vgg11bn_2' : torchmodels.vgg11_bn(pretrained=pretrn),
            'FP_torch_vgg11bn_3' : torchmodels.vgg11_bn(pretrained=pretrn),
            'FP_torch_vgg11bn_4' : torchmodels.vgg11_bn(pretrained=pretrn),
            'FP_torch_vgg13bn' : torchmodels.vgg13_bn(pretrained=pretrn),
            'FP_torch_vgg16bn' : torchmodels.vgg16_bn(pretrained=pretrn),
            'FP_torch_vgg19bn' : torchmodels.vgg19_bn(pretrained=pretrn),
            'FP_torch_resnet18_2' : torchmodels.resnet18(pretrained=pretrn),
            'FP_torch_resnet18_3' : torchmodels.resnet18(pretrained=pretrn),
            'FP_torch_resnet18_4' :torchmodels.resnet18(pretrained=pretrn),
        }

        self.quantization = []
        self.models = []
        for model_type in self.model_types:
            forward_preprocess = self.get_forward_quantization(model_type, device)
            self.quantization.append(forward_preprocess)

        # If the number of classes < 1000 i.e not imageNet
        # use custom models, else use Pytorch's models
        # saves us GPU time to train baseline imageNet
        print("Backpropable Ensemble")
        for quant, model_type in zip(self.quantization, self.model_types):
            model = self.get_bpda_wrapped_model(model_type, quant)
            model.eval()
            self.models.append(model)

        self.num_models = len(self.models)

    def get_bpda_wrapped_model(self, model_name, quantization):
        # if(self.num_classes >= 1000):
        #     # self.model1 = models.resnet18().to(device)
        #     raise ValueError("No ImageNet support")

        preprocess_with_bpda = BPDAWrapper(quantization.forward, forwardsub=quantization.back_approx)

        # See prototype design pattern https://en.wikipedia.org/wiki/Prototype_pattern
        # This dictionary holds the prototypes
        prototype = self.model_prototype_dict[model_name]
        clone =  copy.deepcopy(prototype).to(self.device)

        # Load the corresponding saved model
        clone.load_state_dict(torch.load(self.model_path_dict[model_name], map_location='cuda:0'))
        print("Loaded Model: " + self.model_path_dict[model_name])
        model = nn.Sequential(preprocess_with_bpda, clone)
        return model
        
    def get_forward_quantization(self, q_trans, device):
        if q_trans.lower()=='ht' or q_trans.lower()=='hta1':
            Q = Halftone2d(nin=3).to(device)
        elif q_trans.lower()=='q1' or q_trans.lower() == 'q1a1':
            Q = Quantise2d(n_bits=1).to(device)
        elif q_trans.lower()=='q2' or q_trans.lower() == 'q2a1':
            Q = Quantise2d(n_bits=2).to(device)
        elif q_trans.lower()=='q4':
            Q = Quantise2d(n_bits=4).to(device)
        elif q_trans.lower()=='q6':
            Q = Quantise2d(n_bits=6).to(device)
        elif q_trans.lower()=='q8':
            Q = Quantise2d(n_bits=8).to(device)
        else:
            Q = Quantise2d(n_bits=1, quantise=False).to(device)

        return Q

    def forward(self, inputs):
        out = torch.zeros([inputs.shape[0], self.num_classes]).to(self.device)
        for model in self.models:
            model_output = model(inputs)
            out += model_output
        return out