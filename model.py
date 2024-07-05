import torch
import numpy as np
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
from torch.autograd import Variable
from cbam import *

    
class Resnet18_AttEN(nn.Module):
    def __init__(self, class_num=8, pretrained=True):
        super(Resnet18_AttEN, self).__init__()
        self.f = nn.ModuleList()
        self.g = None
        self.attention_maps = []
        self.num_channel = {
                       'layer1':64, 
                       'layer2':128,
                       'layer3':256,
                       'layer4':512
                      }
        counter = 1
        pass_count = 0
        for name, module in models.resnet18(pretrained=pretrained).named_children():
            if isinstance(module, nn.Linear):
                continue
            if isinstance(module, nn.Sequential):
                self.f.append(module)
                if (pass_count < 1):
                    pass_count += 1
                    counter += 1
                    continue
                self.f.append(CBAM(self.num_channel['layer'+str(counter)], 16))
                counter += 1
                pass_count += 1
            else:
                self.f.append(module)
                
        self.g = nn.Linear(512, class_num, bias=True)
        
    def forward(self, x, if_prompt=False):
        x_attention_feature = []
        inverse_x_attention_feature = []
        attention_map_list = []
        inverse_attention_map_list = []
        for layer in self.f:
            torch.cuda.empty_cache()
            if isinstance(layer, CBAM):
                x, attention_map, inverse_x, inverse_attention_map = layer(x)
                if (if_prompt):
                    x_attention_feature.append(x)
                    inverse_x_attention_feature.append(inverse_x)
                    attention_map_list.append(attention_map)
                    inverse_attention_map_list.append(inverse_attention_map)
                else:
                    x_attention_feature.append(torch.flatten(x, start_dim=1))
                    inverse_x_attention_feature.append(torch.flatten(inverse_x, start_dim=1))
                    attention_map_list.append(attention_map)
                    inverse_attention_map_list.append(inverse_attention_map)
            else:
                x = layer(x)
        
        final_feature = torch.flatten(x, start_dim=1)
        final_out = self.g(final_feature)

        return final_out, x_attention_feature, inverse_x_attention_feature, attention_map_list, inverse_attention_map_list
    
      
class VGG11_AttEN(nn.Module):
    def __init__(self, class_num=8, pretrained=True):
        super(VGG11_AttEN, self).__init__()
        self.f = nn.ModuleList()
        self.g = None
        self.attention_maps = []
        exit_branch_pos = [5, 10, 15, 20]#
        self.num_channel = {
                       'layer1':128, 
                       'layer2':256,
                       'layer3':512,
                       'layer4':512
                      }
        counter = 1
        for name, module in models.vgg11(pretrained=pretrained).features.named_children():
            if int(name) in exit_branch_pos:
                self.f.append(module)
                self.f.append(CBAM(self.num_channel['layer'+str(counter)], 16))
                counter += 1
            else:
                self.f.append(module)
                
        self.f.append(nn.AdaptiveAvgPool2d(output_size=(7, 7)))
        self.f.append(nn.Flatten())
        
        for name, module in models.vgg11(pretrained=True).classifier.named_children():
            if name != '6': #if not the last layer
                self.f.append(module)
            else:           #for last layer
                self.g = nn.Linear(4096, class_num, bias=True)
        
    def forward(self, x, if_prompt=False):
        x_attention_feature = []
        inverse_x_attention_feature = []
        attention_map_list = []
        inverse_attention_map_list = []
        for layer in self.f:
            torch.cuda.empty_cache()
            if isinstance(layer, CBAM):
                x, attention_map, inverse_x, inverse_attention_map = layer(x)
                if (if_prompt):
                    x_attention_feature.append(x)
                    inverse_x_attention_feature.append(inverse_x)
                    attention_map_list.append(attention_map)
                    inverse_attention_map_list.append(inverse_attention_map)
                else:
                    x_attention_feature.append(torch.flatten(x, start_dim=1))
                    inverse_x_attention_feature.append(torch.flatten(inverse_x, start_dim=1))
                    attention_map_list.append(attention_map)
                    inverse_attention_map_list.append(inverse_attention_map)
            else:
                x = layer(x)
                # print(x.shape)
        final_feature = torch.flatten(x, start_dim=1)
        final_out = self.g(final_feature)

        return final_out, x_attention_feature, inverse_x_attention_feature, attention_map_list, inverse_attention_map_list
    
    
    
