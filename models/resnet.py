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
    binarized_tensor = torch.where(input_tensor <= 0, torch.tensor(0.1), torch.tensor(1))
    return binarized_tensor

def activation_shaping_hook(Mt=None, random=False):
        
        def hook_fn(module, input, output):
            #new_output = output * torch.rand_like(output)
            if CONFIG.experiment in ['ASM']:
                p=0.7
                #with probability p assing 0 or 1 to the mask and then multiply it position-wise for the output tensor 
                M_rand=torch.where(torch.rand_like(output) < p, 0.0, 1.0) 
                M=M_rand
                new_output = binarize(output) * M

            elif CONFIG.experiment in ['DA']:

                M=binarize(Mt)
                new_output = binarize(output) * M

            elif CONFIG.experiment in ['BA1']:
                
                new_output = output * Mt

            elif CONFIG.experiment in ['BA2']:
                k=5
                M=binarize(Mt)
                #topk_values, topk_indices = torch.topk(output, k)
                topk_values, topk_indices = torch.topk(output.view(-1), k)
                mask = torch.zeros_like(M.view(-1))
                mask[topk_indices] = 1.0
                mask = mask.view(M.shape)
                #mask = torch.zeros_like(M)
                #mask[topk_indices] = 1.0
                M=M*mask
                new_output = output * M
            
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
    
    def initialize_hooks(self, M=None,penultimate=True):
        # To register the forward hooks --

        if penultimate:
            # Access the penultimate layer (before GAP) in ResNet18
            target_layer = self.resnet.layer4[0].bn1#list(self.resnet.modules())[-7]  # Access the specific layer
            random = CONFIG.experiment in ['ASM']
            # Register forward hook on the penultimate layer
            hook=target_layer.register_forward_hook(activation_shaping_hook(M,random))
            self.hooks.append(hook)
        else:
            i=0
            for module in self.resnet.modules():
                if isinstance(module, nn.Conv2d):
                    hook = module.register_forward_hook(activation_shaping_hook(M[i]))
                    i+=1
                    self.hooks.append(hook)

    def get_activation(self,input_data, penultimate=True):
        activations = []
        hooks=[]

        def hook_fn(module, input, output):
            activations.append( output.detach())

        if penultimate:
            target_layer =self.resnet.layer4[0].bn1
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

        if CONFIG.experiment in ['DA']:
            if x_targ is not None:
                Mt=self.get_activation(x_targ)
                self.initialize_hooks(Mt)

        elif CONFIG.experiment in ['ASM']:
            self.initialize_hooks()

        return self.resnet(x)
    

        # def forward(self, x):
        #     return self.resnet(x)

######################################################
