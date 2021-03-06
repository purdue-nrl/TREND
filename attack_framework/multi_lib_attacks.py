import torch
import torch.nn as nn
import foolbox
import numpy as np
import advertorch
from attack_framework.attacks import LinfPGDAttack_with_normalization, LinfRandAttack_with_normalization, Linf_MIFGSM_Attack_with_normalization
from advertorch.utils import NormalizeByChannelMeanStd
from advertorch.bpda import BPDAWrapper


'''
    This is a LAL (Library Abstraction Layer)
    Wrapper around foolbox, advertorch and custom attack implementation
'''
class attack_wrapper():
    '''
    Example of attack_params
    {
        'lib': 'foolbox',
        'attack':'pgd',
        'iterations': 40,
        'epsilon': 0.01,
        'stepsize': 0.01,
        'bpda': True,
        'random': True
        'preprocess': <pointer to quantize or hafltone or FP>,
        'custom_norm_func': <pointer to custom_3channel_img_normalization_with_dataset_params>,
        'targeted': True,
    }

    Example of dataset_params
    {
        'mean': [0.5, 0.5, 0.5],
        'std': [0.5, 0.5, 0.5],
        'num_classes': 10
    }
    '''
    def __init__(self, model, device, **params):
        self.model = model
        self.device = device
        self.target_model = model

        attack_params = params['attack_params']
        dataset_params = params['dataset_params']
        self.targeted = attack_params['targeted'] 

        # Strategy pattern, see https://en.wikipedia.org/wiki/Strategy_pattern
        init_dict = {
            'advertorch': self.init_advertorch,
            'foolbox':self.init_foolbox,
            'custom':self.init_custom
        }

        lib = attack_params['lib']
        init_func = init_dict[lib]
        self.attack_info = init_func(model, device, attack_params, dataset_params)
        self.attack_params = attack_params
        print('Using attack_wrapper arguments used:')
        print(attack_params)
    '''
    Initialize the objects needed to talk to foolbox library
    '''
    def init_foolbox(self, model, device, attack_params, dataset_params):
        mean = dataset_params['mean']
        std = dataset_params['std']
        self.normalize = attack_params['custom_norm_func'] 

        
        if(attack_params['bpda'] == True):
            # Right way to handle exception in python see https://stackoverflow.com/questions/2052390/manually-raising-throwing-an-exception-in-python
            # Explains all the traps of using exception, does a good job!! I mean the link :)
            raise ValueError("BPDA not supported with foolbox")

        preprocessing = dict(mean=mean, std=std, axis=-3)
        fmodel = foolbox.models.PyTorchModel(model, 
                                             bounds=(0, 1),
                                             device=device,
                                             preprocessing=preprocessing)

        attack_name = attack_params['attack'].lower()
        if(attack_name == 'pgd'):
            iterations = attack_params['iterations']
            stepsize = attack_params['stepsize']
            epsilon = attack_params['epsilon']
            random = attack_params['random']

            attack = foolbox.attacks.LinfProjectedGradientDescentAttack(random_start=random,
                                                                        abs_stepsize=stepsize,
                                                                        steps=iterations)


            # Return attack dictionary
            return {'attack': attack, 
                    'iterations': iterations,
                    'epsilon': epsilon, 
                    'stepsize': stepsize,
                    'random': random,
                    'fmodel': fmodel}

        elif(attack_name == 'cw'):
            iterations = attack_params['iterations']
            epsilon = attack_params['epsilon']
            attack = foolbox.attacks.L2CarliniWagnerAttack(steps=iterations)

            # Return attack dictionary
            return {'attack': attack, 
                    'iterations': iterations,
                    'epsilon': epsilon,
                    'fmodel': fmodel }

        elif(attack_name == 'deepfool'):
            iterations = attack_params['iterations']
            epsilon = attack_params['epsilon']
            attack = foolbox.attacks.LinfDeepFoolAttack(steps = iterations)
            return{'attack': attack, 
                    'iterations': iterations,
                    'epsilon': epsilon,
                    'fmodel': fmodel  }

        else:    
            # Right way to handle exception in python see https://stackoverflow.com/questions/2052390/manually-raising-throwing-an-exception-in-python
            # Explains all the traps of using exception, does a good job!! I mean the link :)
            raise ValueError("Unsupported attack")
    
    '''
    Initialize the objects needed to talk to advetorch library
    '''
    def init_advertorch(self, model, device, attack_params, dataset_params):
        mean = dataset_params['mean']
        std = dataset_params['std']
        num_classes = dataset_params['num_classes']

        self.normalize = NormalizeByChannelMeanStd(mean=mean, std=std)
        basic_model = model

        if(attack_params['bpda'] == True):
            preprocess = attack_params['preprocess']
            preprocess_bpda_wrapper = BPDAWrapper(preprocess, forwardsub=preprocess.back_approx)
            attack_model = nn.Sequential(self.normalize, preprocess_bpda_wrapper, basic_model).to(device)
        else:
            attack_model = nn.Sequential(self.normalize, basic_model).to(device)

        attack_name = attack_params['attack'].lower()
        if(attack_name == 'pgd'):
            iterations = attack_params['iterations']
            stepsize = attack_params['stepsize']
            epsilon = attack_params['epsilon']
            attack = advertorch.attacks.LinfPGDAttack
            random = attack_params['random']
            # Return attack dictionary
            return {'attack': attack, 
                    'iterations': iterations,
                    'epsilon': epsilon, 
                    'stepsize': stepsize,
                    'model': attack_model,
                    'random': random }
        elif(attack_name == 'l2pgd'):
            iterations = attack_params['iterations']
            stepsize = attack_params['stepsize']
            epsilon = attack_params['epsilon']
            attack = advertorch.attacks.L2PGDAttack
            random = attack_params['random']
            # Return attack dictionary
            return {'attack': attack, 
                    'iterations': iterations,
                    'epsilon': epsilon, 
                    'stepsize': stepsize,
                    'model': attack_model,
                    'random': random }

        elif(attack_name == 'cw'):
            iterations = attack_params['iterations']
            epsilon = attack_params['epsilon']
            confidence = attack_params['confidence']
            attack = advertorch.attacks.CarliniWagnerL2Attack

            # Return attack dictionary
            return {'attack': attack, 
                    'iterations': iterations,
                    'confidence': confidence,
                    'model': attack_model,
                    'num_classes': num_classes }

        else:    
            # Right way to handle exception in python see https://stackoverflow.com/questions/2052390/manually-raising-throwing-an-exception-in-python
            # Explains all the traps of using exception, does a good job!! I mean the link :)
            raise ValueError("Unsupported attack")

    '''
    Initialize the objects needed to talk to custom library
    '''
    def init_custom(self, model, device, attack_params, dataset_params):
        self.normalize = attack_params['custom_norm_func'] 
        attack_name = attack_params['attack'].lower()
        if(attack_name == 'pgd'):
            iterations = attack_params['iterations']
            stepsize = attack_params['stepsize']
            epsilon = attack_params['epsilon']
            random = attack_params['random']
            attack_obj = LinfPGDAttack_with_normalization(clip_min=0.0, 
                                                          clip_max=1.0,
                                                          epsilon=epsilon,
                                                          k=iterations,
                                                          a=stepsize,
                                                          random_start=random,
                                                          loss_func='xent')

            if(attack_params['bpda'] == True):
                attack = attack_obj.perturb_bpda
            else:
                attack = attack_obj.perturb

            # Return attack dictionary
            return {'attack': attack, 
                    'iterations': iterations,
                    'epsilon': epsilon, 
                    'stepsize': stepsize,
                    'model': attack_obj }

        elif(attack_name == 'randpgd'):
            iterations = attack_params['iterations']
            stepsize = attack_params['stepsize']
            epsilon = attack_params['epsilon']
            random = attack_params['random']
            attack_obj = LinfRandAttack_with_normalization(clip_min=0.0, 
                                                           clip_max=1.0,
                                                           epsilon=epsilon,
                                                           k=iterations,
                                                           a=stepsize)

            attack = attack_obj.perturb_bpda

            # Return attack dictionary
            return {'attack': attack, 
                    'iterations': iterations,
                    'epsilon': epsilon, 
                    'stepsize': stepsize,
                    'model': attack_obj }
        
        elif(attack_name == 'mifgsm'):
            iterations = attack_params['iterations']
            decayFactor = attack_params['stepsize']
            epsilon = attack_params['epsilon']
            random = attack_params['random']
            attack_obj = Linf_MIFGSM_Attack_with_normalization(clip_min=0.0, 
                                                               clip_max=1.0,
                                                               epsilon=epsilon,
                                                               k=iterations,
                                                               mu=decayFactor)

            attack = attack_obj.perturb_bpda

            # Return attack dictionary
            return {'attack': attack, 
                    'iterations': iterations,
                    'epsilon': epsilon, 
                    'stepsize': decayFactor,
                    'model': attack_obj }

        else:    
            # Right way to handle exception in python see https://stackoverflow.com/questions/2052390/manually-raising-throwing-an-exception-in-python
            # Explains all the traps of using exception, does a good job!! I mean the link :)
            raise ValueError("Unsupported attack")
   
    
    # Generate adversarial images as specifed by initialization parameters.
    def generate_adversary(self,
                           x_batch, 
                           labels, 
                           target_class=None, 
                           update_epsilon=None,
                           adv_train_model=None, 
                           targeted=False,
                           image_at_each_step=False):

        x_batch = x_batch.detach().clone()
        labels = labels.detach().clone()

        if(update_epsilon != None):
            self.attack_info['epsilon'] = update_epsilon
                
        lib = self.attack_params['lib']
        attack = self.attack_info['attack']
        if(lib == 'foolbox'):
            data = x_batch
            target = labels
            fmodel = self.attack_info['fmodel']

            if self.targeted:
                target = foolbox.criteria.TargetClass(target_class)
            
            if adv_train_model !=None:
                raise ValueError("Unsupported adv training attack. Each epoch must update the network under attack")

            raw_advs, perturbed_data, success = attack(fmodel,
                                                       data,
                                                       target,
                                                       epsilons=self.attack_info['epsilon'])                                                        

            # Send the data and label to the device
            un_norm_perturbed_data = perturbed_data.to(self.device)
            perturbed_data = self.normalize(un_norm_perturbed_data)
            return perturbed_data, un_norm_perturbed_data

        elif(lib == 'advertorch'):
            if adv_train_model !=None:
                raise ValueError("Unsupported adv training attack. Each epoch must update the network under attack")

            if(self.attack_params['attack'].lower() == 'pgd'):
                adversary = attack(self.attack_info['model'], 
                                   loss_fn = nn.CrossEntropyLoss(reduction="mean"),
                                   eps=self.attack_info['epsilon'],
                                   eps_iter=self.attack_info['stepsize'],
                                   nb_iter=self.attack_info['iterations'],
                                   rand_init=True,
                                   targeted=self.targeted)

            elif(self.attack_params['attack'].lower() == 'l2pgd'):
                adversary = attack(self.attack_info['model'], 
                                   loss_fn = nn.CrossEntropyLoss(reduction="mean"),
                                   eps=self.attack_info['epsilon'],
                                   eps_iter=self.attack_info['stepsize'],
                                   nb_iter=self.attack_info['iterations'],
                                   rand_init=True,
                                   targeted=self.targeted)

            elif(self.attack_params['attack'].lower() == 'cw'):
                    adversary = attack(self.attack_info['model'],
                                       num_classes=self.attack_info['num_classes'],
                                       max_iterations=self.attack_info['iterations'],
                                       confidence=self.attack_info['confidence'],
                                       targeted=self.targeted)

            data = x_batch.to(self.device)
            target = labels.to(self.device)
            un_norm_perturbed_data = adversary.perturb(data, target)
            perturbed_data = self.normalize(un_norm_perturbed_data)
            return perturbed_data, un_norm_perturbed_data
        
        elif(lib == 'custom'):
            data = x_batch.to(self.device)
            target = labels.to(self.device)
            
            if self.targeted ==False:
                if adv_train_model !=None:
                    self.model = adv_train_model
                if update_epsilon != None:
                    self.attack_info['model'].epsilon = update_epsilon

                if(self.attack_params['attack'].lower() == 'pgd'):
                    if(self.attack_params['bpda']):
                        if(image_at_each_step):
                            un_norm_perturbed_data, img_list = attack(data, 
                                                                    target,
                                                                    self.model, 
                                                                    normalization_function=self.normalize,
                                                                    forward=self.attack_params['preprocess'], 
                                                                    backward_replacement=self.attack_params['preprocess'].back_approx,
                                                                    image_at_each_step=image_at_each_step)
                        else:
                            un_norm_perturbed_data = attack(data, 
                                                            target,
                                                            self.model, 
                                                            normalization_function=self.normalize,
                                                            forward=self.attack_params['preprocess'], 
                                                            backward_replacement=self.attack_params['preprocess'].back_approx)                      

                    else:                
                        un_norm_perturbed_data = attack(data, 
                                                        target,
                                                        self.model, 
                                                        normalization_function=self.normalize)
            
                elif(self.attack_params['attack'].lower() == 'randpgd'):
                    un_norm_perturbed_data = attack(data, 
                                target,
                                self.model, 
                                normalization_function=self.normalize,
                                forward=self.attack_params['preprocess']) 

                elif(self.attack_params['attack'].lower() == 'mifgsm'):
                    un_norm_perturbed_data = attack(data, 
                                target,
                                self.model, 
                                normalization_function=self.normalize,
                                forward=self.attack_params['preprocess'])


                else:
                    raise ValueError ("Unsupported targeted attack. custom PGD does not have support for tageted attack")     
                    
                
                perturbed_data = self.normalize(un_norm_perturbed_data)

                if(image_at_each_step):
                    return perturbed_data, un_norm_perturbed_data, img_list

                return perturbed_data, un_norm_perturbed_data
            else:
                 raise ValueError ("Unsupported targeted attack. custom PGD does not have support for tageted attack")
        else:
            # Right way to handle exception in python see https://stackoverflow.com/questions/2052390/manually-raising-throwing-an-exception-in-python
            # Explains all the traps of using exception, does a good job!! I mean the link :)
            raise ValueError("Unsupported attack")
