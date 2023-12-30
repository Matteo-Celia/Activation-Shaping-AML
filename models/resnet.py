import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

class BaseResNet18(nn.Module):
    def __init__(self):
        super(BaseResNet18, self).__init__()
        self.resnet = resnet18(weights=ResNet18_Weights)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 7)

    def forward(self, x):
        return self.resnet(x)

######################################################
# TODO: either define the Activation Shaping Module as a nn.Module
#class ActivationShapingModule(nn.Module):
#...
#
# OR as a function that shall be hooked via 'register_forward_hook'
#def activation_shaping_hook(module, input, output):
#...
def activation_shaping_hook(module, input, output):
    #print(f'{type(module)} - {output.abs().sum()}', end='\t')
    output = output * torch.where(torch.rand_like(output) < 0.5, 0.0, 1.0) 
    #print(f'{output.abs().sum()}')
    return output
#
######################################################
# TODO: modify 'BaseResNet18' including the Activation Shaping Module
class ASHResNet18(nn.Module):
    def __init__(self):
        super(ASHResNet18, self).__init__()
#        ...
#    
#    def forward(self, x):
#        ...
#
######################################################
