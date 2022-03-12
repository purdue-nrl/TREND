import torch
from models.resnet import *
from models.vgg import *
import torchvision.models as models
from utils.halftone import Halftone2d
from utils.quantise import Quantise2d,RandomQuantise2d
import os

def instantiate_model (dataset='cifar10',
                       num_classes=10, 
                       load_model='FP', 
                       q_tf='FP', 
                       arch='resnet',
                       dorefa=False, 
                       abit=32, 
                       wbit=32,
                       qin=False, 
                       qout=False,
                       suffix='', 
                       load=False,
                       torch_weights=False,
                       device='cpu'):

    # Instantiate model1
    # RESNET IMAGENET
    if(arch == 'torch_resnet18'):
        if dorefa:
            raise ValueError ("Dorefa net unsupported for {}".format(arch))
        else:
            model = models.resnet18(pretrained=torch_weights)
            model_name = dataset.lower()+ "_" + load_model + "_" + arch + suffix
    elif(arch == 'torch_resnet34'):
        if dorefa:
            raise ValueError ("Dorefa net unsupported for {}".format(arch))
        else:
            model = models.resnet34(pretrained=torch_weights)
            model_name = dataset.lower()+ "_" + load_model + "_" + arch + suffix
    elif(arch == 'torch_resnet50'):
        if dorefa:
            raise ValueError ("Dorefa net unsupported for {}".format(arch))
        else:
            model = models.resnet50(pretrained=torch_weights)
            model_name = dataset.lower()+ "_" + load_model + "_" + arch + suffix
    elif(arch == 'torch_resnet101'):
        if dorefa:
            raise ValueError ("Dorefa net unsupported for {}".format(arch))
        else:
            model = models.resnet101(pretrained=torch_weights)
            model_name = dataset.lower()+ "_" + load_model + "_" + arch + suffix
    elif(arch == 'torch_resnet152'):
        if dorefa:
            raise ValueError ("Dorefa net unsupported for {}".format(arch))
        else:
            model = models.resnet152(pretrained=torch_weights)
            model_name = dataset.lower()+ "_" + load_model + "_" + arch + suffix
    elif(arch == 'torch_resnet34'):
        if dorefa:
            raise ValueError ("Dorefa net unsupported for {}".format(arch))
        else:
            model = models.resnet34(pretrained=torch_weights)
            model_name = dataset.lower()+ "_" + load_model + "_" + arch + suffix
    elif(arch == 'torch_resnext50_32x4d'):
        if dorefa:
            raise ValueError ("Dorefa net unsupported for {}".format(arch))
        else:
            model = models.resnext50_32x4d(pretrained=torch_weights)
            model_name = dataset.lower()+ "_" + load_model + "_" + arch + suffix
    elif(arch == 'torch_resnext101_32x8d'):
        if dorefa:
            raise ValueError ("Dorefa net unsupported for {}".format(arch))
        else:
            model = models.resnext101_32x8d(pretrained=torch_weights)
            model_name = dataset.lower()+ "_" + load_model + "_" + arch + suffix
    elif(arch == 'torch_wide_resnet50_2'):
        if dorefa:
            raise ValueError ("Dorefa net unsupported for {}".format(arch))
        else:
            model = models.wide_resnet50_2(pretrained=torch_weights)
            model_name = dataset.lower()+ "_" + load_model + "_" + arch + suffix
    elif(arch == 'torch_wide_resnet101_2'):
        if dorefa:
            raise ValueError ("Dorefa net unsupported for {}".format(arch))
        else:
            model = models.wide_resnet101_2(pretrained=torch_weights)
            model_name = dataset.lower()+ "_" + load_model + "_" + arch + suffix
    #VGG IMAGENET
    elif(arch == 'torch_vgg11'):
        if dorefa:
            raise ValueError ("Dorefa net unsupported for {}".format(arch))
        else:
            model = models.vgg11(pretrained=torch_weights)
            model_name = dataset.lower()+ "_" + load_model + "_" + arch + suffix
    elif(arch == 'torch_vgg11bn'):
        if dorefa:
            raise ValueError ("Dorefa net unsupported for {}".format(arch))
        else:
            model = models.vgg11_bn(pretrained=torch_weights)
            model_name = dataset.lower()+ "_" + load_model + "_" + arch + suffix
    elif(arch == 'torch_vgg13'):
        if dorefa:
            raise ValueError ("Dorefa net unsupported for {}".format(arch))
        else:
            model = models.vgg13(pretrained=torch_weights)
            model_name = dataset.lower()+ "_" + load_model + "_" + arch + suffix
    elif(arch == 'torch_vgg13bn'):
        if dorefa:
            raise ValueError ("Dorefa net unsupported for {}".format(arch))
        else:
            model = models.vgg13_bn(pretrained=torch_weights)
            model_name = dataset.lower()+ "_" + load_model + "_" + arch + suffix
    elif(arch == 'torch_vgg16'):
        if dorefa:
            raise ValueError ("Dorefa net unsupported for {}".format(arch))
        else:
            model = models.vgg16(pretrained=torch_weights)
            model_name = dataset.lower()+ "_" + load_model + "_" + arch + suffix
    elif(arch == 'torch_vgg16bn'):
        if dorefa:
            raise ValueError ("Dorefa net unsupported for {}".format(arch))
        else:
            model = models.vgg16_bn(pretrained=torch_weights)
            model_name = dataset.lower()+ "_" + load_model + "_" + arch + suffix
    elif(arch == 'torch_vgg19'):
        if dorefa:
            raise ValueError ("Dorefa net unsupported for {}".format(arch))
        else:
            model = models.vgg19(pretrained=torch_weights)
            model_name = dataset.lower()+ "_" + load_model + "_" + arch + suffix
    elif(arch == 'torch_vgg19bn'):
        if dorefa:
            raise ValueError ("Dorefa net unsupported for {}".format(arch))
        else:
            model = models.vgg19_bn(pretrained=torch_weights)
            model_name = dataset.lower()+ "_" + load_model + "_" + arch + suffix
    #MOBILENET IMAGENET   
    elif(arch == 'torch_mobnet'):
        if dorefa:
            raise ValueError ("Dorefa net unsupported for {}".format(arch))
        else:
            model = models.mobilenet_v2(pretrained=torch_weights)
            model_name = dataset.lower()+ "_" + load_model + "_" + arch + suffix
    #DENSENET IMAGENET
    elif(arch == 'torch_densenet121'):
        if dorefa:
            raise ValueError ("Dorefa net unsupported for {}".format(arch))
        else:
            model = models.densenet121(pretrained=torch_weights)
            model_name = dataset.lower()+ "_" + load_model + "_" + arch + suffix
    elif(arch == 'torch_densenet169'):
        if dorefa:
            raise ValueError ("Dorefa net unsupported for {}".format(arch))
        else:
            model = models.densenet169(pretrained=torch_weights)
            model_name = dataset.lower()+ "_" + load_model + "_" + arch + suffix
    elif(arch == 'torch_densenet201'):
        if dorefa:
            raise ValueError ("Dorefa net unsupported for {}".format(arch))
        else:
            model = models.densenet201(pretrained=torch_weights)
            model_name = dataset.lower()+ "_" + load_model + "_" + arch + suffix
    elif(arch == 'torch_densenet161'):
        if dorefa:
            raise ValueError ("Dorefa net unsupported for {}".format(arch))
        else:
            model = models.densenet161(pretrained=torch_weights)
            model_name = dataset.lower()+ "_" + load_model + "_" + arch + suffix
    #RESNET CIFAR   
    elif(arch == 'resnet' or arch == 'resnet18'  ):
        if dorefa:
            model = ResNet18_Dorefa(num_classes=num_classes, abit=abit, wbit=wbit)
            model_name = dataset.lower()+ "_" + load_model + "_" + arch +"_a" + str(abit) + 'w'+ str(wbit) + suffix
        else:
            model = ResNet18(num_classes=num_classes)
            model_name = dataset.lower()+ "_" + load_model + "_" + arch + suffix
    elif( arch == 'resnet34'  ):
        if dorefa:
            model = ResNet34_Dorefa(num_classes=num_classes, abit=abit, wbit=wbit)
            model_name = dataset.lower()+ "_" + load_model + "_" + arch +"_a" + str(abit) + 'w'+ str(wbit) + suffix
        else:
            model = ResNet34(num_classes=num_classes)
            model_name = dataset.lower()+ "_" + load_model + "_" + arch + suffix
    elif( arch == 'resnet50'  ):
        if dorefa:
            model = ResNet50_Dorefa(num_classes=num_classes, abit=abit, wbit=wbit)
            model_name = dataset.lower()+ "_" + load_model + "_" + arch +"_a" + str(abit) + 'w'+ str(wbit) + suffix
        else:
            model = ResNet50(num_classes=num_classes)
            model_name = dataset.lower()+ "_" + load_model + "_" + arch + suffix
    elif( arch == 'resnet101'  ):
        if dorefa:
            model = ResNet101_Dorefa(num_classes=num_classes, abit=abit, wbit=wbit)
            model_name = dataset.lower()+ "_" + load_model + "_" + arch +"_a" + str(abit) + 'w'+ str(wbit) + suffix
        else:
            model = ResNet101(num_classes=num_classes)
            model_name = dataset.lower()+ "_" + load_model + "_" + arch + suffix
    elif( arch == 'resnet152'  ):
        if dorefa:
            model = ResNet152_Dorefa(num_classes=num_classes, abit=abit, wbit=wbit)
            model_name = dataset.lower()+ "_" + load_model + "_" + arch +"_a" + str(abit) + 'w'+ str(wbit) + suffix
        else:
            model = ResNet152(num_classes=num_classes)
            model_name = dataset.lower()+ "_" + load_model + "_" + arch + suffix
    #VGG CIFAR
    elif(arch[0:3] == 'vgg'):
        len_arch = len(arch)
        if arch[len_arch-2:len_arch]=='bn' and arch[len_arch-4:len_arch-2]=='bn':
            batch_norm_conv=True
            batch_norm_linear=True
            cfg= arch[3:len_arch-4]
        elif arch [len_arch-2: len_arch]=='bn':
            batch_norm_conv=True
            batch_norm_linear=False
            cfg= arch[3:len_arch-2]
        else:
            batch_norm_conv=False
            batch_norm_linear=False
            cfg= arch[3:len_arch]
        if dorefa:
            model = vgg_Dorefa(cfg=cfg, batch_norm_conv=batch_norm_conv, batch_norm_linear=batch_norm_linear ,num_classes=num_classes, a_bit=abit, w_bit=wbit)
            model_name = dataset.lower()+ "_" + load_model + "_" + arch +"_a" + str(abit) + 'w'+ str(wbit) + suffix
            
        else:   
            model = vgg(cfg=cfg, batch_norm_conv=batch_norm_conv, batch_norm_linear=batch_norm_linear ,num_classes=num_classes)
            model_name = dataset.lower()+ "_" + load_model + "_" + arch + suffix
    else:
        # Right way to handle exception in python see https://stackoverflow.com/questions/2052390/manually-raising-throwing-an-exception-in-python
        # Explains all the traps of using exception, does a good job!! I mean the link :)
        raise ValueError("Unsupported neural net architecture")
    model = model.to(device)
    
    #Select the input transformation
    q_trans=q_tf.lower()[0:2]
    if q_trans=='ht':
        Q = Halftone2d(nin=3).to(device)
    elif q_trans=='q1':
        Q = Quantise2d(n_bits=1).to(device)
    elif q_trans=='q2':
        Q = Quantise2d(n_bits=2).to(device)
    elif q_trans=='q4':
        Q = Quantise2d(n_bits=4).to(device)
    elif q_trans=='q6':
        Q = Quantise2d(n_bits=6).to(device)
    elif q_trans=='q8':
        Q = Quantise2d(n_bits=8).to(device)
    elif q_trans=='fp':
        Q = Quantise2d(n_bits=1,quantise=False).to(device)
    elif q_trans=='m':
        Q = RandomQuantise2d(device)
    else:
        raise ValueError
    if load == True and torch_weights == False :
        print(" Using Model: " + arch)
        model_path = os.path.join('./pretrained/', dataset.lower(),  model_name + '.ckpt')
        model.load_state_dict(torch.load(model_path, map_location='cuda:0'))
        print(' Loaded trained model from :' + model_path)
        print(' {}'.format(Q))
    
    else:
        model_path = os.path.join('./pretrained/', dataset.lower(),  model_name + '.ckpt')
        print(' Training model save at:' + model_path)
    print('')
    return model, model_name, Q
