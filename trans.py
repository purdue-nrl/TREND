import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse
import eagerpy as ep
import foolbox
from utils.load_dataset import load_dataset
from foolbox.utils import accuracy, samples
from attack_framework.attacks import LinfPGDAttack_with_normalization
from utils.halftone import Halftone2d
from utils.quantise import Quantise2d
from utils.instantiate_model import instantiate_model
from models.resnet import *
from utils.inference import *
from models.ensemble import Ensemble
from attack_framework.multi_lib_attacks import attack_wrapper
from utils.str2bool import str2bool
from utils.instantiate_model import *
import os 

parser = argparse.ArgumentParser(description='Test transferability', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--dataset',          default='CIFAR10',     type=str,          help='Set dataset to use')
parser.add_argument('--device',           default=0,             type=int,          help='individual Device')
parser.add_argument('--parallel',         default=False,         type=str2bool,     help='Device in  parallel')
parser.add_argument('--train_batch_size', default=1024,          type=int,          help='Train batch size')
parser.add_argument('--test_batch_size',  default=1024,          type=int,          help='Test batch size')
parser.add_argument('--attack',           default='PGD',         type=str,          help='Type of attack [PGD, CW]')
parser.add_argument('--visualize',        default=False,         type=str2bool,     help='Visualize adversarial images')
parser.add_argument('--use_lib',          default='custom',      type=str,          help='Use [foolbox, advtorch, custom] code for adversarial attack')
parser.add_argument('--use_bpda',         default=True,          type=str2bool,     help='Use Backward Pass through Differential Approximation when using attack')
parser.add_argument('--iterations',       default=40,            type=int,          help='Number of iterations of PGD')
parser.add_argument('--stepsize',         default=0.01,          type=float,        help='stepsize for PGD')
parser.add_argument('--num_steps',        default=0,             type=int,          help='Number of epsilons for PGD')
parser.add_argument('--conf',             default=0,             type=float,        help='Confidence control for CW attack')
parser.add_argument('--save_path',        default='./outputs/',  type=str,          help='Save path for the output file')

parser.add_argument('--load_model',       default='FP',          type=str,          help='Quantization transfer function-Q1 Q2 Q4 Q6 Q8 HT FP')
parser.add_argument('--arch_load',        default='resnet',      type=str,          help='Network architecture resnet, CIFARconv')
parser.add_argument('--torch_weights',    default=False,         type=str2bool,     help='Use weights from torch trained model')
parser.add_argument('--dorefa_load',      default=False,         type=str2bool,     help='Use Dorefa Net')
parser.add_argument('--qout_load',        default=False,         type=str2bool,     help='Output layer weight quantisation')
parser.add_argument('--qin_load',         default=False,         type=str2bool,     help='Input layer weight quantisation')
parser.add_argument('--abit_load',        default=32,            type=int,          help='activation quantisation precision')
parser.add_argument('--wbit_load',        default=32,            type=int,          help='Weight quantisation precision')
parser.add_argument('--suffix_load',      default='',            type=str,          help='Suffix of load model')
parser.add_argument('--opt_src',          default='',            type=str,          help='Default is SGD, use \'_adam\' for Adam ')
parser.add_argument('--opt_tgt',          default='',            type=str,          help='Default is SGD, use \'_adam\' for Adam ')

parser.add_argument('--q_tf',             default='all',         type=str,          help='Quantization transfer function-Q1 Q2 Q4 Q6 Q8 HT FP')
parser.add_argument('--q_number',         default=0,             type=int,          help='if q_tf=all set 0=FP 1=Q1 2=Q2 3=Q4 4=Q6 5=Q8 6=HT')
parser.add_argument('--dorefa_test',      default=False,         type=str2bool,     help='Use Dorefa Net')
parser.add_argument('--qout_test',        default=False,         type=str2bool,     help='Output layer weight quantisation')
parser.add_argument('--qin_test',         default=False,         type=str2bool,     help='Input layer weight quantisation')
parser.add_argument('--abit_test',        default=1,             type=int,          help='activation quantisation precision')
parser.add_argument('--wbit_test',        default=32,            type=int,          help='Weight quantisation precision')
parser.add_argument('--random_start',     default=True,          type=str2bool,     help='Random start for adv attack')
parser.add_argument('--analysis',         default='arch',        type=str     ,     help='Analysis     type, Arch, Input Quant, Weight Quant or Activation Quant')

global args
args = parser.parse_args()

args.save_path = os.path.join(args.save_path, args.dataset.lower(), 'transferability/')

print(args)
if args.analysis.lower() != 'seed':
    args.save_path = args.save_path + args.attack + "_" + args.analysis
else:
    args.save_path = args.save_path + args.attack + "_"

args.attack = args.attack.lower()
args.suffix_load = args.opt_src + args.suffix_load

if(args.attack.lower() == "cw"):
    args.save_path += str(int(args.conf)) + "_"

if args.attack.lower() == "fgsm":
    args.attack ="pgd"
    args.iterations = 1
    args.stepsize = 1.0

if args.load_model=='A1':
    args.load_model='FP'
    args.dorefa_load=True
    args.abit=1
    args.wbit=32

class Ensemble(nn.Module):
    def __init__(self,
                 device,
                 models, 
                 num_classes):
        super(Ensemble, self).__init__()
        print("Loading ensemble models:")
        self.models = []
        self.model_name = []
        self.quantization = []
        
        for precision_type, arch , suffix in models:
            if args.dorefa_test and len(suffix) >= 7:
                
                individual_model, name, q = instantiate_model(dataset=args.dataset,
                                                              load_model=precision_type,
                                                              q_tf=precision_type,
                                                              arch=arch,
                                                              suffix=str(suffix[7:]),
                                                              dorefa=True,
                                                              abit=int(suffix[2:4]),
                                                              wbit=int(suffix[5:7]),
                                                              num_classes=num_classes,
                                                              device=device,
                                                              torch_weights=args.torch_weights,
                                                              load=True)
            else:
                individual_model, name, q = instantiate_model(dataset=args.dataset,
                                                              load_model=precision_type,
                                                              q_tf=precision_type,
                                                              arch=arch,
                                                              suffix=suffix,
                                                              num_classes=num_classes,
                                                              device=device,
                                                              torch_weights=args.torch_weights,
                                                              load=True)

            self.models.append(individual_model.eval())
            self.quantization.append(q)
            self.model_name.append(name)

        self.device = device
        self.num_classes = num_classes
        self.net_type = 'ensemble'
        self.num_models = len(self.models)

    def forward(self, inputs):
        out = []
        for i, ind_model in enumerate(self.models):
            out_ind = ind_model(self.quantization[i](inputs))
            out.append(out_ind)

        return out



#--------------------------------------------------
# Parse input arguments
#--------------------------------------------------

# Parameters
batch_size    = args.train_batch_size
b_size_test   = args.test_batch_size

# Setup right device to run on
if args.parallel:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
else:
    device = torch.device('cuda'":" + str(args.device) if torch.cuda.is_available() else 'cpu')

train_loader, val_loader, test_loader, normalization_function, unnormalization_function, num_classes, mean, std, img_dim = load_dataset(dataset=args.dataset, 
                                                                                                                                       train_batch_size=batch_size,
                                                                                                                                       test_batch_size=b_size_test, 
                                                                                                                                       device=device)

# Instantiate model
source_net, model_name, Q_src = instantiate_model(dataset=args.dataset,
                                                  num_classes=num_classes,
                                                  load_model = args.load_model,
                                                  q_tf=args.load_model, 
                                                  arch=args.arch_load,
                                                  dorefa=args.dorefa_load,
                                                  abit=args.abit_load, 
                                                  wbit=args.wbit_load, 
                                                  qin=args.qin_load,
                                                  qout=args.qout_load,
                                                  suffix=args.suffix_load,
                                                  torch_weights=args.torch_weights,
                                                  load=True,
                                                  device=device)


if(args.dataset.lower() == 'cifar10' or args.dataset.lower() == 'cifar100'):
    if(args.analysis.lower() == 'arch'):
        models = [('FP','resnet', args.suffix_load),
                  ('FP','resnet34', args.suffix_load),
                  ('FP','resnet50', args.suffix_load),
                  ('FP','resnet101', args.suffix_load),  
                  ('FP','vgg11', args.suffix_load),
                  ('FP','vgg19', args.suffix_load),            
                  ('FP','vgg11bn', args.suffix_load),
                  ('FP','vgg19bn', args.suffix_load)]
    elif(args.analysis.lower() == 'seed'):
        models = [('FP', args.arch_load, args.opt_tgt + '_1'),
                  ('FP', args.arch_load, args.opt_tgt + '_2'),
                  ('FP', args.arch_load, args.opt_tgt + '_3'),
                  ('FP', args.arch_load, args.opt_tgt + '_4'),
                  ('FP', args.arch_load, args.opt_tgt + '_5'),
                  ('FP', args.arch_load, args.opt_tgt + '_6'),
                  ('FP', args.arch_load, args.opt_tgt + '_7'),
                  ('FP', args.arch_load, args.opt_tgt + '_8'),
                  ('FP', args.arch_load, args.opt_tgt + '_9'),
                  ('FP', args.arch_load, args.opt_tgt + '_10')]
    elif(args.analysis.lower() == 'quant'):
        models = [('HT','resnet', args.suffix_load),
                  ('Q1','resnet', args.suffix_load),
                  ('Q2','resnet', args.suffix_load),
                  ('Q4','resnet', args.suffix_load),
                  ('Q6','resnet', args.suffix_load),
                  ('Q8','resnet', args.suffix_load),
                  ('FP','resnet', args.suffix_load)]
    elif(args.analysis.lower() == 'act'):
        models = [('FP','resnet', '_a01w32'+args.suffix_load),
                  ('FP','resnet', '_a02w32'+args.suffix_load),
                  ('FP','resnet', '_a04w32'+args.suffix_load),
                  ('FP','resnet', '_a08w32'+args.suffix_load),
                  ('FP','resnet', '_a16w32'+args.suffix_load),
                  ('FP','resnet', args.suffix_load)]
    elif(args.analysis.lower() == 'weight'):
        models = [('FP','resnet', '_a32w01'+args.suffix_load),
                  ('FP','resnet', '_a32w02'+args.suffix_load),
                  ('FP','resnet', '_a32w04'+args.suffix_load),
                  ('FP','resnet', '_a32w08'+args.suffix_load),
                  ('FP','resnet', '_a32w16'+args.suffix_load),
                  ('FP','resnet', args.suffix_load)]
else:
    if(args.analysis.lower() == 'arch'):
        args.torch_weights=True
        models = [('FP','torch_resnet18', ''),
                  ('FP','torch_resnet34', ''),
                  ('FP','torch_resnet101', ''),  
                  ('FP','torch_vgg11', ''),
                  ('FP','torch_vgg19', ''),
                  ('FP','torch_densenet121', ''),
                  ('FP','torch_wide_resnet50_2', '')]
  
    elif(args.analysis.lower() == 'quant'):
        models = [('HT','torch_resnet18', ''),
                  ('Q1','torch_resnet18', ''),
                  ('Q2','torch_resnet18', ''),
                  ('Q4','torch_resnet18', ''),
                  ('Q6','torch_resnet18', ''),
                  ('Q8','torch_resnet18', ''),
                  ('FP','torch_resnet18', '')]
    else:
        raise ValueError("Unsupported")
 

target_net = Ensemble(device=device, models=models, num_classes=num_classes)
test_name = 'all'    
print('Warning: Not using DataParallel')

target_net = target_net.to(device)
target_net.eval()
source_net.eval()

attack_params = {   'lib': args.use_lib,
                    'attack': args.attack,
                    'iterations': args.iterations,
                    'epsilon': 0.03,
                    'stepsize': args.stepsize,
                    'confidence': args.conf,
                    'bpda': args.use_bpda,
                    'preprocess': Q_src,
                    'custom_norm_func': normalization_function,
                    'random': False,
                    'targeted': False }

dataset_params =  { 'mean': mean,
                    'std': std,
                    'num_classes': num_classes }

params = {'attack_params': attack_params,
          'dataset_params': dataset_params}

attack = attack_wrapper(source_net, device, **params)

iterations = args.iterations

def test(epsilon):
    correct = 0
    total = 0
    norm_total= 0
    gen_adv_imgs = 0
    L2 = 0
    Linf = 0
    num_trans_imgs = torch.zeros((target_net.num_models, 1), device=device)
    num_gen_imgs = torch.zeros((target_net.num_models, 1), device=device)
    conf = torch.zeros((target_net.num_models, 1), device=device)

    for batch_idx, (data, label) in enumerate(test_loader):
        data = data.to(device)
        label = label.to(device)

        # Generate adversaries on the source network
        perturbed_data, un_norm_perturbed_data = attack.generate_adversary(data, label, update_epsilon=epsilon)

        with torch.no_grad():
            # Re-classify the perturbed image
            output = source_net(Q_src(perturbed_data))  

        # Get the index of the max log-probability
        final_pred = output.max(1, keepdim=True)[1]
        src_adv = final_pred.squeeze(dim=1)

        total += data.shape[0]

        with torch.no_grad():
            output = source_net(Q_src(normalization_function(data)))

        # Get the index of the max log-probability
        final_pred = output.max(1, keepdim=True)[1]
        correct += (final_pred == label).sum()
        src_clean = final_pred.squeeze(dim=1)
        norm_total+= torch.sum(torch.where ((src_clean == label) & (src_adv != label), torch.ones_like(label), torch.zeros_like(label)))
        L2 += torch.sum(torch.where((src_clean == label) & (src_adv != label),torch.norm(data - un_norm_perturbed_data, p=2, dim=(1,2,3)), torch.zeros_like(label).float() ))
        Linf += torch.sum(torch.where((src_clean == label) & (src_adv != label) , torch.norm(data - un_norm_perturbed_data, p=float('inf'), dim=(1,2,3)), torch.zeros_like(label).float() ) )


        # Was previously correctly classifed by the source model and was then incorrectly classfifed by the target
        adv = torch.where((src_adv != label) & (src_clean == label), torch.ones_like(label), torch.zeros_like(label))
        current_adv_images = torch.sum(adv)
        gen_adv_imgs += current_adv_images

        with torch.no_grad():
            target_adv = target_net(perturbed_data)
            target_clean = target_net(normalization_function(data))

        for i in range(target_net.num_models):
            trgt_adv = target_adv[i].max(1, keepdim=True)[1].squeeze(-1)
            trgt_clean = target_clean[i].max(1, keepdim=True)[1].squeeze(-1)

            transferred = torch.where ((trgt_clean == label) & (trgt_adv != label) & (src_adv != label ) & (src_clean == label),
                                       torch.ones_like(label),
                                       torch.zeros_like(label))

            # Number of the adversarial images to the source such that the clean version of the images
            # are correctly classified correctly by both the source and target models
            common_images = torch.where ((trgt_clean == label) & (src_adv != label) & (src_clean == label),
                                         torch.ones_like(label),
                                         torch.zeros_like(label))

            num_gen_imgs[i] += torch.sum(common_images)

            num_trans_imgs[i] += torch.sum(transferred)

            prob = torch.nn.functional.softmax(target_adv[i], dim=1).max(1)[0]
            conf[i] += torch.where ((trgt_clean == label) & (trgt_adv != label) & (src_adv != label ) & (src_clean == label),
                                    prob,
                                    torch.zeros_like(label).float()).sum()

        
        # norm_2 = float(L2.item()) / float(total)
        # norm_inf = float(Linf.item()) / float(total)
            
    # Calculate final accuracy for this epsilon
    final_acc = float(correct) / float(total)
    norm_2 = float(L2.item()) / num_gen_imgs.max() # .max Corresponds to generated on the source
    norm_inf = float(Linf.item()) / num_gen_imgs.max()

    print("L2 {:.6f} Linf {:.6f}".format(norm_2, norm_inf))

    # Return the accuracy and an adversarial example
    return final_acc, num_trans_imgs, num_gen_imgs, conf, norm_2, norm_inf

if(args.num_steps > 0):
    num_steps = args.num_steps + 1
    step = 0.1 / num_steps
    epsilons = np.arange(step, 0.1, step)
else:
    epsilons = np.array([8.0/255.0])
    if args.attack == "l2pgd":
        epsilons = np.array([0.145])

num_adv_images = []
num_trans_images = []
percent = []
l2 = []
linf = []

# Run test for each epsilon
for eps in epsilons:
    torch.cuda.empty_cache()
    _, num_trans_imgs, common_gen_adv_images, conf, L2, Linf = test(eps)
    num_adv_images.append(common_gen_adv_images)
    num_trans_images.append(num_trans_imgs.view((num_trans_imgs.shape[0],1)))
    percent.append(num_trans_imgs / common_gen_adv_images)
    l2.append(L2)
    linf.append(Linf)

if args.opt_tgt != args.opt_src:
    if args.opt_tgt =="" :
        file_name = model_name + args.opt_tgt + "_src.pt"
    elif args.opt_src =="" :
        file_name = model_name + args.opt_tgt + "_tgt.pt"
else:
    file_name = model_name + ".pt"

print (torch.cat(percent, dim=1))
print(common_gen_adv_images)
conf = conf / num_trans_images[0]
print("src adv conf {:.4f}, target adv conf {:.4f}".format(conf[0][0], conf[1:].mean()))
print("Trans mean {:.2f} std {:.2f}".format(num_trans_images[0][1:].mean(), num_trans_images[0][1:].std()))
print("Gen mean {:.2f} std {:.2f}".format(common_gen_adv_images[1:].mean(), common_gen_adv_images[1:].std()))
print(conf)

data = {'num_adv_images': common_gen_adv_images,
        'num_trans_images': num_trans_images,
        'percent': percent,
        'L2': l2,
        'Linf': linf,
        'conf_src': conf[0][0],
        'conf_tgt': conf[1:],
        'epsilons': epsilons}

torch.save(data, args.save_path + file_name)