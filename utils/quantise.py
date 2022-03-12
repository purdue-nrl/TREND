import torch
import torch.nn as nn
from utils.halftone import *

class Quantise2d(nn.Module):
    def __init__(self, n_bits=1, quantise=True, un_normalized=False):
        
        super(Quantise2d, self).__init__()
        assert n_bits <= 8.0
        self.n_bits = n_bits
        self.quantise=quantise
        self.un_normalized=un_normalized
        
    
    def forward(self, data):
        if self.quantise:
            if self.un_normalized and self.n_bits<8:

                data = (data * 255.0).long().float()
                bin_width = 255.0/2**(self.n_bits)
                data=torch.where(data==255.0,data-0.5 * bin_width * torch.ones_like(data), data)
                data = torch.floor( data/bin_width ) * bin_width + 0.5 * bin_width * torch.ones_like(data)
                data = data / 255.0

            elif(not(self.un_normalized)) :

                bin_width = (data.max()-data.min())/ 2**self.n_bits
                data=torch.where(data==data.max(),data-0.5*bin_width,data)
                data = torch.floor( (data - torch.ones_like(data)*data.min() )/ bin_width)  * bin_width  + torch.ones_like(data) * (0.5 * bin_width + data.min())

            else:

                raise ValueError            
        return data
    
    def back_approx(self, x):
        return x

    def __repr__(self):
        if(self.quantise):
            return "Quantize2d, bits:" + str(self.n_bits)
        else:
            return "Quantize2d FP"

    def __str__(self):
        if(self.quantise):
            return "Quantize2d, bits:" + str(self.n_bits)
        else:
            return "Quantize2d FP"
    

class RandomQuantise2d(nn.Module):
    def __init__(self, device):
        Q1 = Quantise2d(n_bits=1).to(device)
        Q2 = Quantise2d(n_bits=2).to(device)
        HT = Halftone2d(nin=3).to(device)
        FP = Quantise2d(n_bits=1,quantise=False).to(device)
        self.list = [Q1, Q2, HT, FP]
        super(RandomQuantise2d, self).__init__()
    
    def forward(self, data):
        quant = self.list [torch.LongTensor(1).random_(4)]     
        return quant(data)
    
    def back_approx(self, x):
        return x    

    