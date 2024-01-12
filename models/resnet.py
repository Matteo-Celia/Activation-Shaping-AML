import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
from globals import CONFIG

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
def binarize(input_tensor):
    binarized_tensor = torch.where(input_tensor <= 0, torch.tensor(0), torch.tensor(1))
    return binarized_tensor

def activation_shaping_hook(Mt, random=False):
        
        def hook_fn(module, input, output):
            #new_output = output * torch.rand_like(output)
            if random:
                p=0.7
                #with probability p assing 0 or 1 to the mask and then multiply it position-wise for the output tensor 
                M_rand=torch.where(torch.rand_like(output) < p, 0.0, 1.0) 
                M=M_rand
            else:

                M=binarize(Mt)
                #dont binarize output too maybe
            new_output = binarize(output) * M
            new_output=new_output.to(torch.float32)
            return new_output
        
        return hook_fn


#
######################################################
# TODO: modify 'BaseResNet18' including the Activation Shaping Module
class ASHResNet18(nn.Module):
    def __init__(self):
        super(ASHResNet18, self).__init__()
        self.resnet = resnet18(weights=ResNet18_Weights)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 7)

        self.hooks = []
    
    def initialize_hooks(self, M,penultimate=True):
        # To register the forward hooks --

        if penultimate:
            # Access the penultimate layer (before GAP) in ResNet18
            penultimate_layer = list(self.resnet.children())[-3]  # Access the specific layer

            # Register forward hook on the penultimate layer
            hook=penultimate_layer.register_forward_hook(activation_shaping_hook(M))
            self.hooks.append(hook)
        else:
            i=0
            for module in self.resnet.modules():
                if isinstance(module, nn.Conv2d):
                    hook = module.register_forward_hook(activation_shaping_hook(M[i]))
                    i+=1
                    self.hooks.append(hook)

    def get_activation(self,input_data, penultimate=False):
        activations = []
        hooks=[]

        def hook_fn(module, input, output):
            activations.append( output.detach())

        if penultimate:
            target_layer = list(self.resnet.children())[-3]
            hook = target_layer.register_forward_hook(hook_fn)
            hooks.append(hook)
        else:
            for module in self.resnet.modules():
                if isinstance(module, nn.Conv2d):
                    hook = module.register_forward_hook(hook_fn)
                    hooks.append(hook)
        # Forward pass to capture activations
        self.resnet(input_data)

        # Remove the hooks
        for hook in hooks:
            hook.remove()

        return activations[0] if activations else None
    
    def remove_hooks(self):

        # Remove registered hooks
        for hook in self.hooks:
            hook.remove()
    
    
    def forward(self, x, x_targ=None):
        if x_targ is not None:
            Mt=self.get_activation(x_targ,True)
            self.initialize_hooks(Mt)
        return self.resnet(x)
    

        # def forward(self, x):
        #     return self.resnet(x)

######################################################
