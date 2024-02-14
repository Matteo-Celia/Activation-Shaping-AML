import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
from globals import CONFIG



def binarize(tensor):
    return torch.where(tensor > 0, torch.tensor(1), torch.tensor(0.0))

def register_forward_hooks(model, hook, layer_type, skip_step=None):
    hook_handles = []
    layer_count = 0
    for layer in model.modules():
        if isinstance(layer, layer_type):
            if skip_step is None:
                hook_handles.append(layer.register_forward_hook(hook))
            else:
                if layer_count % skip_step == 0:
                    hook_handles.append(layer.register_forward_hook(hook))
                layer_count += 1
    #print(f'Registered {len(hook_handles)} forward hooks to {layer_type}')
    return hook_handles

def remove_forward_hooks(hook_handles):
    for hook in hook_handles:
        hook.remove()

def asm_hook(module, input, output):
    #print(f"Activation hook triggered for module: {module.__class__.__name__}")
    ratio = 0.50
    p = torch.full_like(output, ratio)
    mask = torch.bernoulli(p)
    mask_bin = binarize(mask)
    output_bin = binarize(output)
    return output_bin * mask_bin

class BaseResNet18(nn.Module):
    def __init__(self):
        super(BaseResNet18, self).__init__()
        self.resnet = resnet18(weights=ResNet18_Weights)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 7)

    def forward(self, x):
        return self.resnet(x)
    
class DAResNet18(nn.Module):
    def __init__(self):
        super(DAResNet18, self).__init__()
        self.resnet = resnet18(weights=ResNet18_Weights)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 7)
        self.actmaps_target = []
        self.forward_turn = 'target'
    
    def forward_target(self, x_target):
        with torch.autocast(device_type=CONFIG.device,enabled=False):
            with torch.no_grad():
                self.forward_turn='target'
                self.resnet(x_target)

    def forward(self, x_source, x_target=None):
        # unregister other forward hooks
        # register forward hooks
        # unregister forward hooks
        with torch.autocast(device_type=CONFIG.device,enabled=False):

            #if x_target is not None:
            #    self.forward_turn = 'target'
                
                #x=self.resnet(x_target)
                #print('x {}'.format(x.requires_grad))                  
            self.forward_turn = 'source'
            print(len(self.actmaps_target),x_source.shape)
            z= self.resnet(x_source)
            print('z {}'.format(z.requires_grad))
            return z
    
    def rec_actmaps_hook(self, module, input, output):
        if self.forward_turn == 'target':
            #print(f"rec_actmaps_hook triggered for module: {module.__class__.__name__}")
            #print(f"actmaps length: {len(self.actmaps_target)}")
            self.actmaps_target.append(output.detach()) #.clone()
    
    def asm_source_hook(self, module, input, output):
            if self.forward_turn == 'source':
                #print(f"asm_source_hook triggered for module: {module.__class__.__name__}")
                #print(f"actmaps length: {len(self.actmaps_target)}")
                mask = self.actmaps_target.pop(0)

                mask_bin = binarize(mask)

                if CONFIG.experiment in ['DA-BA2']:

                    k=100 # hyperparameter to tune
                    #topk_values, topk_indices = torch.topk(output, k)
                    topk_values, topk_indices = torch.topk(output.view(-1), k)
                    mask_flatten = torch.zeros_like(mask_bin.view(-1))
                    mask_flatten[topk_indices] = 1.0
                    mask_topk = mask_flatten.view(mask_bin.shape)
                    # multiply initial mask for topk mask
                    mask_bin=mask_bin*mask_topk
                    
                output_bin = binarize(output)
                output = output_bin * mask_bin
                return output
            
class DGResNet18(nn.Module):
    def __init__(self):
        super(DGResNet18, self).__init__()
        self.resnet = resnet18(weights=ResNet18_Weights)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 7)
        self.actmaps1 = []
        self.actmaps2 = []
        self.actmaps3 = []
        self.rec_turn = 1

    def forward(self, x):
        return self.resnet(x)

    def rec_actmaps(self, x1, x2, x3):
        self.rec_turn = 1
        self.resnet(x1)
        self.rec_turn = 2
        self.resnet(x2)
        self.rec_turn = 3
        self.resnet(x3)

    def rec_actmaps_hook(self, module, input, output):
        if self.rec_turn == 1:
            self.actmaps1.append(output.detach())
        elif self.rec_turn == 2:
            self.actmaps2.append(output.detach())
        elif self.rec_turn == 3:
            self.actmaps3.append(output.detach())
    
    def asm_hook(self, module, input, output):
        mask1 = (self.actmaps1.pop(0) > 0).float()
        mask2 = (self.actmaps2.pop(0) > 0).float()
        mask3 = (self.actmaps3.pop(0) > 0).float()
        output_bin = (output > 0).float()
        mask = torch.cat((mask1, mask2, mask3)) #should be multiplication
        return mask * output_bin



######################################################
# TODO: either define the Activation Shaping Module as a nn.Module
#class ActivationShapingModule(nn.Module):
#...
#
# OR as a function that shall be hooked via 'register_forward_hook'
#def activation_shaping_hook(module, input, output):
#...
#
######################################################
# TODO: modify 'BaseResNet18' including the Activation Shaping Module
#class ASHResNet18(nn.Module):
#    def __init__(self):
#        super(ASHResNet18, self).__init__()
#        ...
#    
#    def forward(self, x):
#        ...
#
######################################################
