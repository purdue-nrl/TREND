import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.dataloader import DataLoader
from attack_framework.custom_normalization_functions import *
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler

def load_dataset(dataset='CIFAR10', train_batch_size=128, test_batch_size=128, val_split=0.0, augment = True, padding_crop = 4, shuffle = True, random_seed=None ,device ='cpu'):
    '''
    Inputs
    dataset -> CIFAR10, CIFAR100, TinyImageNet, ImageNet
    train_batch_size -> batch size for training dataset
    test_batch_size -> batch size for testing dataset
    val_split -> percentage of training data split as validation dataset
    augment -> bool flag for Random horizontal flip and shift with padding
    padding_crop -> units of pixel shift
    shuffle -> bool flag for shuffling the training and testing dataset
    random_seed -> Fixes the shuffle seed for reproducing the results
    device -> cuda device or cpu
    return -> bool for returning the mean, std, img_size
    '''
    # Load dataset
    # Use the following transform for training and testing
    if (dataset.lower() == 'mnist'):
        mean=[0.1307]
        std=[0.3081]
        img_dim = 28
        img_ch = 1
        num_classes=10
        num_worker = 0
        test_transform = transforms.Compose([
                                                transforms.ToTensor()
                                            ])
        
        val_transform = test_transform

        if augment:
            train_transform = transforms.Compose([
                                                transforms.RandomCrop(img_dim, padding= padding_crop),
                                                transforms.RandomHorizontalFlip(),
                                                transforms.ToTensor(),
                                                ])
        else:
            train_transform = test_transform

        trainset = torchvision.datasets.MNIST(root='./data/MNIST/clean',
                                            train=True,
                                            download=True,
                                            transform=train_transform)
        valset = torchvision.datasets.MNIST(root='./data/MNIST/clean',
                                            train=True,
                                            download=True,
                                            transform=val_transform)

        testset = torchvision.datasets.MNIST(root='./data/MNIST/clean',
                                            train=False,
                                            download=True, 
                                            transform=test_transform)
        
    elif(dataset.lower() == 'cifar10'):
        mean=[0.4914, 0.4822, 0.4465]
        std=[0.2023, 0.1994, 0.2010]
        img_dim = 32
        img_ch = 3
        num_classes=10
        num_worker = 40

        test_transform = transforms.Compose([
                                                transforms.ToTensor()
                                            ])
        
        val_transform = test_transform

        if augment:
            train_transform = transforms.Compose([
                                                transforms.RandomCrop(img_dim, padding= padding_crop),
                                                transforms.RandomHorizontalFlip(),
                                                transforms.ToTensor(),
                                                ])
        else:
            train_transform = test_transform

        trainset = torchvision.datasets.CIFAR10(root='./data/CIFAR10/clean',
                                            train=True,
                                            download=True,
                                            transform=train_transform)
        valset = torchvision.datasets.CIFAR10(root='./data/CIFAR10/clean',
                                            train=True,
                                            download=True,
                                            transform=val_transform)

        testset = torchvision.datasets.CIFAR10(root='./data/CIFAR10/clean',
                                            train=False,
                                            download=True, 
                                            transform=test_transform)
        

    elif(dataset.lower() == 'cifar100'):
        mean = [0.5071, 0.4867, 0.4408]
        std = [0.2675, 0.2565, 0.2761]
        img_dim = 32
        img_ch = 3
        num_classes = 100
        num_worker = 40
        
        test_transform = transforms.Compose([
                                                transforms.ToTensor()
                                            ])

        val_transform = test_transform

        if augment:
            train_transform = transforms.Compose([
                                            transforms.RandomCrop(img_dim, padding=padding_crop),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.ToTensor(),
                                            ])
        else:
            train_transform = test_transform


        trainset = torchvision.datasets.CIFAR100(root='./data/CIFAR100/clean',
                                            train=True,
                                            download=True,
                                            transform=train_transform)
        valset = torchvision.datasets.CIFAR100(root='./data/CIFAR100/clean',
                                            train=True,
                                            download=True,
                                            transform=test_transform)
        testset = torchvision.datasets.CIFAR100(root='./data/CIFAR100/clean',
                                            train=False,
                                            download=True, 
                                            transform=test_transform)
       
    elif(dataset.lower() == 'tinyimagenet'):
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        root = './data/TinyImageNet/clean'
        img_dim = 64
        img_ch = 3
        num_classes = 200
        num_worker = 40
        
        test_transform = transforms.Compose([
                                                transforms.ToTensor(),
                                            ])
        val_transform = test_transform

        if augment:
            train_transform = transforms.Compose([
                                            transforms.RandomResizedCrop(img_dim),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.ToTensor()
                                        ])
        else:
            train_transform= test_transform

        trainset = TinyImageNet(root=root + '/val/', train_root=root + '/train/', transform=test_transform, val=False) 
        valset =  TinyImageNet(root=root + '/val/', train_root=root + '/train/', transform=test_transform, val=False) 
        testset = TinyImageNet(root=root + '/val/', train_root=root + '/train/', transform=test_transform, val=True)  
        

    elif(dataset.lower() == 'imagenet'):
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        img_dim = 224
        img_ch = 3
        num_classes = 1000
        num_worker = 40
        datapath ='/local/a/imagenet/imagenet2012/'
        #datapath = 'Path for image net goes here' # Set path here
                
        test_transform =  transforms.Compose([
                                                transforms.Resize(256),
                                                transforms.CenterCrop(img_dim),
                                                transforms.ToTensor(),
                                            ])
        val_transform = test_transform
        if augment:
            train_transform = transforms.Compose([transforms.Resize(256),
                                            transforms.RandomResizedCrop(img_dim),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.ToTensor()
                                        ])
        else:
            train_transform = test_transform

        trainset = torchvision.datasets.ImageFolder(root=datapath + 'train', transform=train_transform)
        valset =  torchvision.datasets.ImageFolder(root=datapath + 'train', transform=train_transform)
        testset = torchvision.datasets.ImageFolder(root=datapath + 'val', transform=test_transform)
    else:
        # Right way to handle exception in python see https://stackoverflow.com/questions/2052390/manually-raising-throwing-an-exception-in-python
        # Explains all the traps of using exception, does a good job!! I mean the link :)
        raise ValueError("Unsupported dataset")
    
    #split the training dataset into training and validation sets
    print('\nForming the sampler for train and validation split')
    num_train = len(trainset)
    ind = list (range (num_train))
    split = int ( np.floor(val_split * num_train) )

    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(ind)
    
    train_idx, val_idx =ind[split:], ind[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    val_sampler = SubsetRandomSampler(val_idx)

    # Load dataloader
    print('Loading data to the dataloader \n')
    train_loader = torch.utils.data.DataLoader(trainset, 
                                               batch_size=train_batch_size,
                                               sampler=train_sampler, 
                                               pin_memory=True,
                                               num_workers=num_worker)

    val_loader =  torch.utils.data.DataLoader(valset,
                                              batch_size=train_batch_size,
                                              sampler=val_sampler,
                                              pin_memory=True,
                                              num_workers=num_worker)

    test_loader = torch.utils.data.DataLoader(testset,
                                              batch_size=test_batch_size,
                                              shuffle=False,
                                              pin_memory=True,
                                              num_workers=num_worker)

    if img_ch == 1:
        _stats = {'mean': mean,
            'std': std}
        normalization_params = {**_stats, **{'img_dimensions': [img_ch, img_dim, img_dim]}}
        normalization_function = custom_1channel_img_normalization_with_dataset_params(**normalization_params, device=device)
        unnormalization_function = custom_1channel_img_UnNormalize_with_dataset_params(**normalization_params, device=device)
    elif img_ch ==3 :            
        _stats = {'mean': mean,
            'std': std}
        normalization_params = {**_stats, **{'img_dimensions': [img_ch, img_dim, img_dim]}}
        normalization_function = custom_3channel_img_normalization_with_dataset_params(**normalization_params, device=device)
        unnormalization_function = custom_3channel_img_UnNormalize_with_dataset_params(**normalization_params, device=device)
    else:
        raise ValueError
    return train_loader, val_loader, test_loader, normalization_function, unnormalization_function, num_classes, mean, std, img_dim
