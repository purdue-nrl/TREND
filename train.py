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
parser.add_argument('--epochs',                 default=400,            type=int,       help='Set number of epochs')
parser.add_argument('--dataset',                default='CIFAR10',      type=str,       help='Set dataset to use')
parser.add_argument('--device',                 default=0,              type=int,       help='individual Device')
parser.add_argument('--parallel',               default=False,          type=str2bool,  help='Device in  parallel')
parser.add_argument('--lr',                     default=0.01,           type=float,     help='Learning Rate')
parser.add_argument('--test_accuracy_display',  default=True,           type=str2bool,  help='Test after each epoch')
parser.add_argument('--optimizer',              default='SGD',          type=str,       help='Optimizer for training')
parser.add_argument('--loss',                   default='crossentropy', type=str,       help='loss function for training')
parser.add_argument('--resume',                 default=False,          type=str2bool,  help='resume training from a saved checkpoint')
parser.add_argument('--include_validation',     default=False,          type=str2bool,  help='retrains with validation set')

# Dataloader args
parser.add_argument('--train_batch_size',       default=512,            type=int,       help='Train batch size')
parser.add_argument('--test_batch_size',        default=512,            type=int,       help='Test batch size')
parser.add_argument('--val_split',              default=0.1,            type=float,     help='fraction of training dataset split as validation')
parser.add_argument('--augment',                default=True,           type=str2bool,  help='Random horizontal flip and random crop')
parser.add_argument('--padding_crop',           default=4,              type=int,       help='Padding for random crop')
parser.add_argument('--shuffle',                default=True,           type=str2bool,  help='Shuffle the training dataset')
parser.add_argument('--random_seed',            default=0,              type=int,       help='Initialising the seed for reproducibility')

# Model parameters
parser.add_argument('--save_seed',              default=False,          type=str2bool,  help='Save the seed')
parser.add_argument('--use_seed',               default=False,          type=str2bool,  help='For Random initialisation')
parser.add_argument('--load_model',             default='FP',           type=str,       help='Quantization transfer function-Q1 Q2 Q4 Q6 Q8 HT FP')
parser.add_argument('--suffix',                 default='',             type=str,       help='appended to model name')
parser.add_argument('--dorefa',                 default=False,          type=str2bool,  help='Use Dorefa Net')
parser.add_argument('--arch',                   default='resnet',       type=str,       help='Network architecture')
parser.add_argument('--qout',                   default=False,          type=str2bool,  help='Output layer weight quantisation')
parser.add_argument('--qin',                    default=False,          type=str2bool,  help='Input layer weight quantisation')
parser.add_argument('--abit',                   default=32,             type=int,       help='Activation quantisation precision')
parser.add_argument('--wbit',                   default=32,             type=int,       help='Weight quantisation precision')

#attack parameters
parser.add_argument('--adv_trn',                default=False,          type=str2bool,  help='adv Training')
parser.add_argument('--attack',                 default='PGD',          type=str,       help='Type of attack [PGD, CW]')
parser.add_argument('--lib',                    default='custom',       type=str,       help='Use [foolbox, advtorch, custom] code for adversarial attack')
parser.add_argument('--use_bpda',               default=True,           type=str2bool,  help='Use Backward Pass through Differential Approximation when using attack')
parser.add_argument('--random',                 default=True,           type=str2bool,  help='Random seed/strating points')
parser.add_argument('--iterations',             default=40,             type=int,       help='Number of iterations of PGD')
parser.add_argument('--epsilon',                default=0.031,          type=float,     help='epsilon for PGD')
parser.add_argument('--stepsize',               default=0.01,           type=float,     help='stepsize for attack')
global args
args = parser.parse_args()
print(args)

# Parameters
num_epochs    = args.epochs
batch_size    = args.train_batch_size
b_size_test   = args.test_batch_size
learning_rate = args.lr

# Setup right device to run on
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Use the following transform for training and testing
print('\n')
train_loader, val_loader,test_loader, normalization_function, unnormalization_function,num_classes, mean, std, img_dim = load_dataset(dataset=args.dataset, 
                                                                                                                                      train_batch_size = batch_size,
                                                                                                                                      test_batch_size=b_size_test, 
                                                                                                                                      val_split=args.val_split, 
                                                                                                                                      augment = args.augment, 
                                                                                                                                      padding_crop = args.padding_crop, 
                                                                                                                                      shuffle = args.shuffle, 
                                                                                                                                      random_seed=args.random_seed ,
                                                                                                                                      device=device)

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
                                       suffix=args.suffix, 
                                       device=device)

    
if args.use_seed:  
    if args.save_seed:
        print("Saving Seed")
        torch.save(net.state_dict(),'./seed/'+args.dataset.lower()+'_'+args.arch+".Seed")
    else:
        print("Loading Seed")
        net.load_state_dict(torch.load('./seed/'+args.dataset.lower()+'_'+args.arch+".Seed"))
else:
    print("Random Initialisation")
    
def transform_labels(labels, onehot=True):
    if onehot:
        labels_onehot = torch.FloatTensor(labels.shape[0],num_classes).to(device)
        labels_onehot.zero_()
        labels_onehot.scatter_(1, labels.unsqueeze(1), 1)
        return labels_onehot
    else:
        return labels

# Optimizer
if args.optimizer.lower()=='sgd':
    optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate,momentum=0.9,weight_decay=5e-4)
elif args.optimizer.lower()=='adagrad':
    optimizer = torch.optim.Adagrad(net.parameters(), lr=learning_rate)
elif args.optimizer.lower()=='adam':
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
else:
    raise ValueError ("Unsupported Optimizer")

if args.loss.lower() == 'crossentropy':
     criterion = torch.nn.CrossEntropyLoss()
     onehot=False
elif args.loss.lower() == 'mse':
    criterion=torch.nn.MSELoss()
    onehot=True
else:
    raise ValueError ("Unsupported loss function")

scheduler=torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                               milestones=[int(0.6*args.epochs), int(0.8*args.epochs)],
                                               gamma=0.1)

#setup adv training attack
if args.adv_trn:
    print ('Adversarial Training')
    attack_params = {   'lib': args.lib,
                        'attack': args.attack,
                        'iterations': args.iterations,
                        'epsilon': args.epsilon,
                        'stepsize': args.stepsize,
                        'bpda': args.use_bpda,
                        'preprocess': Q,
                        'custom_norm_func': normalization_function,
                        'targeted': False,
                        'random': args.random }

    dataset_params =  { 'mean': mean,
                        'std': std,
                        'num_classes': num_classes }

    params = {'attack_params': attack_params,
            'dataset_params': dataset_params}
    attack = attack_wrapper(net, device, **params)
    iterations = args.iterations
    model_name += '_adv'
    args.suffix += '_adv'

if args.resume:
    try:
        saved_training_state = torch.load('./pretrained/'+ args.dataset.lower()+'/temp/' + model_name  + '.temp')
        start_epoch =  saved_training_state['epoch']
        optimizer.load_state_dict(saved_training_state['optimizer'])
        net.load_state_dict( saved_training_state['model'])
        best_val_accuracy = saved_training_state['best_val_accuracy']
        best_val_loss = saved_training_state['best_val_loss']
    except:
        start_epoch=0
        best_val_accuracy = 0.0
        best_val_loss = float('inf')
        net.load_state_dict(torch.load('./pretrained/'+ args.dataset.lower()+'/' + model_name + '.ckpt'))
        net=net.to(device)
else:
    start_epoch=0
    best_val_accuracy = 0.0
    best_val_loss = float('inf')
if args.parallel:
    net = nn.DataParallel(net, device_ids=[0,1,2,3])
else:
    net = net.to(device)  
# Train model

for epoch in range(start_epoch,num_epochs,1):
    net.train()
    train_correct = 0.0
    train_total =0.0
    save_ckpt = False
    print('')
    for batch_idx, (data, labels) in enumerate(train_loader):
        data = data.to(device)
        labels = labels.to(device)
        # Generate adversarial image
        if args.adv_trn:
            perturbed_data, un_norm_perturbed_data = attack.generate_adversary(data, labels, adv_train_model = net )
            data = Q(perturbed_data).to(device)
        else:
            data = Q(normalization_function( data )).to(device)
        
        # Clears gradients of all the parameter tensors
        optimizer.zero_grad()
        out = net(data)
        loss = criterion(out, transform_labels(labels, onehot=onehot) )
        loss.backward()
        optimizer.step()
        if batch_idx % 48 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, train_total, (1-args.val_split)* len(train_loader.dataset),
                100. * train_total / ( (1-args.val_split) * len(train_loader.dataset) ), loss.item()))
        train_correct += (out.max(-1)[1] == labels).sum().long().item()
        train_total += labels.size()[0]
    train_accuracy = float(train_correct) * 100.0/float(train_total)
    print('Train Epoch: {} Accuracy : {}/{} [ {:.2f}%)]\tLoss: {:.6f}'.format(
                epoch, train_correct, train_total,train_accuracy, loss.item()))
    
    # Step the scheduler by 1 after each epoch
    scheduler.step()
    
    if args.val_split > 0.0: 
        val_correct, val_total, val_accuracy = inference(Q=Q, normalization_function = normalization_function, net=net, data_loader = val_loader, device=device)
        if val_accuracy >= best_val_accuracy:
            best_val_accuracy = val_accuracy 
            best_val_loss = best_val_loss
            max_epoch = epoch+1 
            save_ckpt = True
    else: 
        val_accuracy= float('inf')
        if (epoch+1)%10==0:
            save_ckpt=True

    if args.parallel:
        saved_training_state={  'epoch'     : epoch+1,
                                'optimizer' : optimizer.state_dict(),
                                'model'     : net.module.state_dict(),
                                'best_val_accuracy' : best_val_accuracy,
                                'best_val_loss' : best_val_loss
                            }
    else:
        saved_training_state={  'epoch'     : epoch+1,
                                'optimizer' : optimizer.state_dict(),
                                'model'     : net.state_dict(),
                                'best_val_accuracy' : best_val_accuracy,
                                'best_val_loss' : best_val_loss
                            }
    torch.save(saved_training_state, './pretrained/'+ args.dataset.lower()+'/temp/' + model_name  + '.temp')
    
    if save_ckpt:
        print("Saving checkpoint...")
        if args.parallel:
            torch.save(net.module.state_dict(), './pretrained/'+ args.dataset.lower()+'/' + model_name  + '.ckpt')
        else:
            torch.save(net.state_dict(), './pretrained/'+ args.dataset.lower()+'/' + model_name + '.ckpt')
        if args.test_accuracy_display:
            # Test model
            # Set the model to eval mode
            test_correct, test_total, test_accuracy = inference(Q=Q, normalization_function = normalization_function, net=net, data_loader = test_loader, device=device)
            print(' Training set accuracy: {}/{}({:.2f}%) \n Validation set accuracy: {}/{}({:.2f}%)\n Test set: Accuracy: {}/{} ({:.2f}%)'.format(
                train_correct,train_total, train_accuracy,
                val_correct,val_total, val_accuracy,
                test_correct, test_total,test_accuracy))
# Test model
# Set the model to eval mode
print("\nEnd of training without reusing Validation set")
if args.val_split > 0.0:
    print('Loading the best model on validation set')
    net.load_state_dict(torch.load('./pretrained/'+ args.dataset.lower()+'/' + model_name + '.ckpt'))
    net=net.to(device)

    val_correct, val_total, val_accuracy = inference(Q=Q, normalization_function = normalization_function, net=net, data_loader = val_loader, device=device)
    print(' Validation set: Accuracy: {}/{} ({:.2f}%)'.format(
                val_correct, val_total, val_accuracy))
else:
    print('Saving the final model')
    if args.parallel:
        torch.save(net.module.state_dict(), './pretrained/'+ args.dataset.lower()+'/' + model_name  + '.ckpt')
    else:
        torch.save(net.state_dict(), './pretrained/'+ args.dataset.lower()+'/' + model_name + '.ckpt')

test_correct, test_total, test_accuracy = inference(Q=Q, normalization_function = normalization_function, net=net, data_loader = test_loader, device=device)
print(' Test set: Accuracy: {}/{} ({:.2f}%)'.format(
            test_correct, test_total, test_accuracy))

train_correct, train_total, train_accuracy = inference(Q=Q, normalization_function = normalization_function, net=net, data_loader = train_loader, device=device)
print(' Train set: Accuracy: {}/{} ({:.2f}%)'.format(
            train_correct, train_total, train_accuracy))

if args.include_validation:
    max_epoch = int( float(max_epoch) * float( len(train_loader)) / float(len(train_loader)+len(val_loader)) )
    train_loader, val_loader,test_loader, normalization_function, unnormalization_function,num_classes, mean, std, img_dim = load_dataset( dataset=args.dataset, 
                                                                                                                    train_batch_size = batch_size,
                                                                                                                    test_batch_size=b_size_test, 
                                                                                                                    val_split= 0.0, 
                                                                                                                    augment = args.augment, 
                                                                                                                    padding_crop = args.padding_crop, 
                                                                                                                    shuffle = args.shuffle, 
                                                                                                                    random_seed=args.random_seed ,
                                                                                                                    device=device)



    net, model_name, Q = instantiate_model(dataset=args.dataset,num_classes = num_classes,load_model = args.load_model,q_tf = args.load_model, arch=args.arch, dorefa=args.dorefa, abit=args.abit, 
                                            wbit=args.wbit, qin=args.qin, qout=args.qout,suffix=args.suffix, device=device)
    print('Retrain to include validation set')
    print('Number of epochs:{}'.format( max_epoch ) )
    # Optimizer
    if args.optimizer.lower()=='sgd':
        optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate,momentum=0.9,weight_decay=5e-4)
    elif args.optimizer.lower()=='adagrad':
        optimizer = torch.optim.Adagrad(net.parameters(), lr=learning_rate)
    elif args.optimizer.lower()=='adam':
        optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    else:
        raise ValueError ("Unsupported Optimizer")

    if args.loss.lower() == 'crossentropy':
        criterion = torch.nn.CrossEntropyLoss()
        onehot=False
    elif args.loss.lower() == 'mse':
        criterion=torch.nn.MSELoss()
        onehot=True
    else:
        raise ValueError ("Unsupported loss function")

    scheduler=torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=[int(0.6*args.epochs), int(0.8*args.epochs)],gamma=0.1)
    
    for epoch in range(max_epoch ):
        net.train()
        train_correct = 0.0
        train_total =0.0
        save_ckpt = False
        print('')
        for batch_idx, (data, labels) in enumerate(train_loader):
            data = data.to(device)
            labels = labels.to(device)
            #generate adversarial image
            if args.adv_trn:
                perturbed_data, un_norm_perturbed_data = attack.generate_adversary(data, labels, adv_train_model = net )
                data = Q(perturbed_data).to(device)
            else:
                data = Q(normalization_function( data )).to(device)
            
            # Clears gradients of all the parameter tensors
            optimizer.zero_grad()
            out = net(data)
            loss = criterion(out, transform_labels(labels, onehot=onehot) )
            loss.backward()
            optimizer.step()
            if batch_idx % 48 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, train_total, (1-args.val_split)* len(train_loader.dataset),
                    100. * train_total / ( (1-args.val_split) * len(train_loader.dataset) ), loss.item()))
            train_correct += (out.max(-1)[1] == labels).sum().long().item()
            train_total += labels.size()[0]

        # Step the scheduler by 1 after each epoch
        scheduler.step()        
                
        train_accuracy = float(train_correct) * 100.0/float(train_total)
        print('Train Epoch: {} Accuracy : {}/{} [ {:.2f}%)]\tLoss: {:.6f}'.format(
                    epoch, train_correct, train_total,train_accuracy, loss.item()))
        if (epoch+1)%10==0:
            save_ckpt=True
        
        if save_ckpt:
            print("Saving checkpoint...")
            if args.parallel:
                torch.save(net.module.state_dict(), './pretrained/'+ args.dataset.lower()+'/' + model_name  + '.ckpt')
            else:
                torch.save(net.state_dict(), './pretrained/'+ args.dataset.lower()+'/' + model_name + '.ckpt')
            if args.test_accuracy_display:
                # Test model
                # Set the model to eval mode
                test_correct, test_total, test_accuracy = inference(Q=Q, normalization_function = normalization_function, net=net, data_loader = test_loader, device=device)
                print(' Training set accuracy: {}/{}({:.2f}%) \n Test set: Accuracy: {}/{} ({:.2f}%)'.format(
                    train_correct,train_total, train_accuracy,
                    test_correct, test_total,test_accuracy))

    print("\nEnd of training with Validation set\nSaving the final model")
    if args.parallel:
        torch.save(net.module.state_dict(), './pretrained/'+ args.dataset.lower()+'/' + model_name  + '.ckpt')
    else:
        torch.save(net.state_dict(), './pretrained/'+ args.dataset.lower()+'/' + model_name + '.ckpt')


    test_correct, test_total, test_accuracy = inference(Q=Q, normalization_function = normalization_function, net=net, data_loader = test_loader, device=device)
    print(' Test set: Accuracy: {}/{} ({:.2f}%)'.format(
                test_correct, test_total, test_accuracy))

    train_correct, train_total, train_accuracy = inference(Q=Q, normalization_function = normalization_function, net=net, data_loader = train_loader, device=device)
    print(' Train set: Accuracy: {}/{} ({:.2f}%)'.format(
                train_correct, train_total,train_accuracy) )
