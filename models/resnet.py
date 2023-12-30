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
        
        #new_output = output * torch.rand_like(output)

        #with probability 0.5 assing 0 or 1 to the mask and then multiply it position-wise for the output tensor 
        new_output = output * torch.where(torch.rand_like(output) < 0.5, 0.0, 1.0) 
        
        return new_output
#
######################################################
# TODO: modify 'BaseResNet18' including the Activation Shaping Module
class ASHResNet18(nn.Module):
    def __init__(self):
        super(ASHResNet18, self).__init__()
        self.resnet = resnet18(weights=ResNet18_Weights)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 7)

        self.hooks = []
        self.initialize_hooks()
    
    def initialize_hooks(self):
        # To register the forward hooks --
        for module in self.resnet.modules():
            if isinstance(module, nn.Conv2d):
                hook = module.register_forward_hook(activation_shaping_hook)
                self.hooks.append(hook)

    def remove_hooks(self):

        # Remove registered hooks
        for hook in self.hooks:
            hook.remove()

    def forward(self, x):
        return self.resnet(x)

######################################################
