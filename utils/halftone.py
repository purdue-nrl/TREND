import torch
import torch.nn as nn

class Halftone2d(nn.Module):
    def __init__(self, nin,bias=1e-3,Normalise=True):
        
        super(Halftone2d, self).__init__()
        self.depthwise = nn.Conv2d(nin, nin, kernel_size=3, padding=1, groups=nin,bias=False)
        self.initialize(nin=nin)
        self.bias=bias
        self.Normalise=Normalise
        
    def forward(self, x):
        if self.Normalise:
            x = ((x.permute(1,2,3,0)-x.mean(-1).mean(-1).mean(-1))/(x.max(-1)[0].max(-1)[0].max(-1)[0]-x.min(-1)[0].min(-1)[0].min(-1)[0])).permute(3,0,1,2).contiguous()
        out = 0.5*torch.sign(self.depthwise(x)+self.bias)+0.5
        return out

    def initialize(self,nin=3,A = torch.tensor([[[[-1.0,-2.0,-1.0], [-2,16,-2], [-1,-2,-1]]]])):
        self.depthwise.trainable=False
        self.depthwise.weight.data = A.repeat((nin,1,1,1))
        
    def back_approx(self, x):
        if self.Normalise:
           x = ((x.permute(1,2,3,0)-x.mean(-1).mean(-1).mean(-1))/(x.max(-1)[0].max(-1)[0].max(-1)[0]-x.min(-1)[0].min(-1)[0].min(-1)[0])).permute(3,0,1,2).contiguous()
        out = 0.5*torch.sigmoid(self.depthwise(x)+self.bias)+0.5

        return out

    def __repr__(self):
        return "Halftone 2d"

    def __str__(self):
        return "Halftone 2d"

