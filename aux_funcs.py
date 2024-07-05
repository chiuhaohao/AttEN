# aux_funcs.py
# contains auxiliary functions for optimizers, internal classifiers, confusion metric
# conversion between CNNs and SDNs and also plotting

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import os
import random
import os.path
import torch.optim as optim
import sys
import itertools as it
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
from model import *

def get_pytorch_device():
    device = 'cpu'
    cuda = torch.cuda.is_available()
    print('Using PyTorch version:', torch.__version__, 'CUDA:', cuda)
    if cuda:
        device = 'cuda'
    return device

# to log the output of the experiments to a file
class Logger(object):
    def __init__(self, log_file, mode='out'):
        if mode == 'out':
            self.terminal = sys.stdout
        else:
            self.terminal = sys.stderr

        self.log = open('{}.{}'.format(log_file, mode), "a")

    def write(self, message):
        self.terminal.write(message)
        self.terminal.flush()
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def __del__(self):
        self.log.close()

def set_logger(log_file):
    sys.stdout = Logger(log_file, 'out')

# the learning rate scheduler

def get_random_seed():
    return 121 # 121 and  or 120(new epochs)

def get_subsets(input_list, sset_size):
    return list(it.combinations(input_list, sset_size))

def set_random_seeds():
    torch.manual_seed(get_random_seed())
    np.random.seed(get_random_seed())
    random.seed(get_random_seed())

def extend_lists(list1, list2, items):
    list1.append(items[0])
    list2.append(items[1])

def get_dataset(dataset, batch_size=128, add_trigger=False):
    if dataset == 'isic2019':
        return load_ISIC2019(batch_size)


def load_ISIC2019(batch_size):
    isic2019 = ISIC2019(batch_size=batch_size)
    return isic2019


def get_output_relative_depths(model):
    total_depth = model.init_depth
    output_depths = []

    for layer in model.layers:
        total_depth += layer.depth

        if layer.no_output == False:
            output_depths.append(total_depth)

    total_depth += model.end_depth

    #output_depths.append(total_depth)
 
    return np.array(output_depths)/total_depth, total_depth


def create_path(path):
    if not os.path.exists(path):
        os.makedirs(path)


def model_exists(models_path, model_name):
    return os.path.isdir(models_path+'/'+model_name)

def get_nth_occurance_index(input_list, n):
    if n == -1:
        return len(input_list) - 1
    else:
        return [i for i, n in enumerate(input_list) if n == 1][n]

def get_lr(optimizers):
    if isinstance(optimizers, dict):
        return optimizers[list(optimizers.keys())[-1]].param_groups[-1]['lr']
    else:
        return optimizers.param_groups[-1]['lr']


def get_pytorch_device():
    device = 'cpu'
    cuda = torch.cuda.is_available()
    print('Using PyTorch version:', torch.__version__, 'CUDA:', cuda)
    if cuda:
        device = 'cuda'
    return device


def get_loss_criterion(dataset_name=''):
    if dataset_name == '':
        return CrossEntropyLoss().cuda()
    if dataset_name == 'celebA':
        return BCEWithLogitsLoss().cuda()
    
def get_sdn(model=''):
    """

    Args:
        model (str, optional): _description_. Defaults to ''.

    Returns:
        nn.module: selected models 
    """
    if model == 'VGG_Early_Exits':
        return VGG_Early_Exits
    if model == 'ResNet_Early_Exits':
        return ResNet_Early_Exits
    if model == 'ResNet18_Early_Exits_CelebA':
        return ResNet18_Early_Exits_CelebA
    if model == 'ResNet18_Early_Exits':
        return ResNet18_Early_Exits
    
    

