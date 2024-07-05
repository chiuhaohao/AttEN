import torch
import time
import data
import argparse
import numpy as np
import torch.nn as nn
import torch.optim as optim
import aux_funcs as af
from SNNL import *
from util.custom_loss import *
from data import *
from tqdm import tqdm
from model import *
from util.bce_acc import *
from sklearn import metrics
import torch.nn.functional as F
from fairness_metric import *
from torchvision import models
from datetime import datetime
import torch.autograd as autograd
from torch.utils.tensorboard import SummaryWriter

def cnn_test(model, loader, device='cpu'):

    model.eval()
    gender_list = []
    label_list = []
    y_pred_list = []
    groupAcc=[]
    with torch.no_grad():
        for batch in tqdm(loader):
            b_x = batch[0].to(device)
            b_y = batch[1].to(device)
            b_mask = batch[2].to(device)
            gender  = batch[3].to(device)
            output, attention_feature, inverse_attention_feature, attention_map_list, inverse_attention_map_list = model(b_x, False)
            _, pred = torch.max(output, 1)
            
            label_list.append(b_y.detach().cpu().numpy())
            y_pred_list.append(pred.detach().cpu().numpy())
            gender_list.append(gender.cpu().numpy())
    label_list = np.concatenate(label_list)
    y_pred_list = np.concatenate(y_pred_list)
    gender_list = np.concatenate(gender_list)
    fairness_metrics = compute_fairness_metrics(label_list, y_pred_list, gender_list)

    for k, v in fairness_metrics.items():
        print('{}:{:.4f}'.format(k, v))

def cnn_train(model, data, epochs, optimizer, scheduler, device='cuda', tensor_board_path='', models_path='', args=None):
    
    writer = SummaryWriter(tensor_board_path)
    SNNL = SoftNearestNeighborLoss()
    sam_prompt = args.prompt
    for epoch in range(1, epochs):     
        CE_loss = [] 
        target_SNN_loss = []
        sensitive_SNN_loss = []
        label_list = []
        y_pred_list = []
        sensitive_group_list = []
        cur_lr = af.get_lr(optimizer)      
        train_loader = data.train_loader
        start_time = time.time()
        model.train()
        print('Epoch: {}/{}'.format(epoch, epochs))
        print('Cur lr: {}'.format(cur_lr))
        target_snnl = 0
        sensitive_snnl = 0
        alpha = args.alpha
        for x, y, b_mask, sensitive_group in tqdm(train_loader): #, mask
            b_x = x.to(device)   # batch x
            b_y = y.to(device)   # batch y
            b_mask = b_mask.to(device)
            b_sensitive_group = sensitive_group.to(device)
            # sam_prompt = False
            output, attention_feature, inverse_attention_feature, attention_map_list, inverse_attention_map_list = model(b_x, sam_prompt)  # cnn final output
            
            new_attention_feature = []
            new_inverse_attention_feature = []
            
            # Augment the feature by the prompt attention map from SAM's mask

            if (sam_prompt):
                for idx, residual_map in enumerate(attention_map_list):
                    # prompt attention map = original attention map + guid mask map 
                    h, w = residual_map.shape[2], residual_map.shape[3]
                    prompt_mask = F.adaptive_max_pool2d(b_mask, (h, w))
                    channel_mean = torch.mean(prompt_mask[:, 1, :, :], dim=(1, 2), keepdim=True)
                    new_prompt_msk = torch.zeros(residual_map.shape[0], 1, h, w).to(device)
                    new_prompt_msk[:, 0, :, :] = channel_mean
                    residual_map_copy = residual_map.clone()
                    residual_map_copy.add_(new_prompt_msk)
                    
                    prompt_attention_feature = residual_map_copy * attention_feature[idx]
                    new_attention_feature.append(torch.flatten(prompt_attention_feature, start_dim=1))
                
                
                for idx, residual_map in enumerate(inverse_attention_map_list):
                    # print(residual_map)
                    inverse_attention_feature[idx] = residual_map * inverse_attention_feature[idx]
                    new_inverse_attention_feature.append(torch.flatten(inverse_attention_feature[idx], start_dim=1))
                    
                attention_feature = new_attention_feature
                inverse_attention_feature = new_inverse_attention_feature
            
            target_snnl = t_id_snnl = s_id_snnl = sensitive_snnl = 0
            
            # v1
            for idx, residual_feature in enumerate(attention_feature):
                t_id_snnl = SNNL(residual_feature, b_y)
                if (torch.isnan(t_id_snnl) or torch.isinf(t_id_snnl)):
                    continue
                target_snnl += t_id_snnl 
            
            # Here for SNNL
            for idx, residual_feature in enumerate(attention_feature):
                t_id_snnl = SNNL(residual_feature, b_y)
                if (torch.isnan(t_id_snnl) or torch.isinf(t_id_snnl)):
                    continue
                target_snnl += t_id_snnl 
                    
            for idx, residual_feature in enumerate(inverse_attention_feature):
                for label in range(args.class_num):
                    mask = b_y == label
                    s_id_snnl = SNNL(residual_feature[mask], b_y[mask])
                    if (torch.isnan(s_id_snnl) or torch.isinf(s_id_snnl)):
                        continue
                    sensitive_snnl += s_id_snnl # sensitive_snnl need to be maximized
            
            _, preds = torch.max(output, 1) 
            
            # tune the target, sensitive weight base on their loss
            criterion = af.get_loss_criterion('')
            loss = criterion(output, b_y)
            if (args.class_num == 8):
                sensitive_weight = 5
                target_weight = 1
            elif (args.class_num == 114):
                sensitive_weight = 0.1
                target_weight = 1
            else:
                sensitive_weight = 1
                target_weight = 1
            loss = loss + alpha*(target_weight*target_snnl - sensitive_weight*sensitive_snnl)
            
            optimizer.zero_grad()           
            loss.backward()
            # loss.mean().backward()        
            optimizer.step()   

            CE_loss.append(loss.mean())
            target_SNN_loss.append(target_snnl.mean()) #
            sensitive_SNN_loss.append(sensitive_snnl.mean())
            
            label_list.append(b_y.detach().cpu().numpy())
            y_pred_list.append(preds.detach().cpu().numpy())
            sensitive_group_list.append(sensitive_group.numpy())

        scheduler.step()
        label_list = np.concatenate(label_list)
        y_pred_list = np.concatenate(y_pred_list)
        sensitive_group_list = np.concatenate(sensitive_group_list)    
        end_time = time.time()


        epoch_time = int(end_time-start_time)

        print('CE Loss: {}'.format(sum(CE_loss) / len(CE_loss)))     
        print('Target SNNL Loss: {}'.format(sum(target_SNN_loss) / len(target_SNN_loss)))  #
        print('Sensitive SNNL Loss: {}'.format(sum(sensitive_SNN_loss) / len(sensitive_SNN_loss)))     
        print('Epoch took {} seconds.'.format(epoch_time))
        writer.add_scalar('CE Loss: ', sum(CE_loss) / len(CE_loss), epoch)
        writer.add_scalar('Target SNNL Loss: ', sum(target_SNN_loss) / len(target_SNN_loss), epoch) #
        writer.add_scalar('Sensitive SNNL Loss: ', sum(sensitive_SNN_loss) / len(sensitive_SNN_loss), epoch)
        writer.add_scalar("Lr/train", cur_lr, epoch)
        print('Start testing...')
        # cnn_test(model, data.vali_loader, device)

        if epoch % 5 == 0:
            torch.save(model, '{}/{}.pth'.format(models_path, epoch))
            cnn_test(model, data.test_loader, device)
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train_sdn')
    parser.add_argument('--training_title', type=str, default='',
                    help='')
    parser.add_argument('--epochs', type=int, default=200,
                    help='')
    parser.add_argument('--lr', type=float, default=0.01,
                    help='')
    parser.add_argument('--batch_size', type=int, default=256,
                    help='')
    parser.add_argument('--dataset', type=str, default='isic2019_mask',
                    help='')
    parser.add_argument('--model', type=str, default='resnet18',
                    help='')
    parser.add_argument('--class_num', type=int, default=8,
                    help='')
    parser.add_argument('--group_num', type=int, default=2,
                    help='')
    parser.add_argument('--image_size', type=int, default=256,
                    help='')
    parser.add_argument('--image_crop_size', type=int, default=224,
                    help='')
    parser.add_argument('--Lambda', type=float, default=5,
                    help='')
    parser.add_argument('--sigma', type=float, default=0.5,
                    help='')
    parser.add_argument('--alpha', type=float, default=0.1,
                    help='')
    parser.add_argument('--prompt', type=bool, default=True,
                    help='')
    parser.add_argument('--split_num', type=int, default=1,
                    help='')
    args = parser.parse_args()
    training_title = args.training_title
    print(training_title)
    random_seed = af.get_random_seed()
    af.set_random_seeds()
    print('Random Seed: {}'.format(random_seed))
    device = af.get_pytorch_device()
    models_path = 'networks/{}/{}'.format(af.get_random_seed(), training_title)
    tensor_board_path = 'runs/{}/train_models{}'.format(training_title, af.get_random_seed())
    af.create_path(models_path)
    af.create_path(tensor_board_path)
    af.create_path('outputs/{}'.format(training_title))
    af.set_logger('outputs/{}/train_models{}'.format(training_title, af.get_random_seed()))

    print("Arguments: ")
    argument_list = ""
    for arg in vars(args):
        argument_list += " --{} {}".format(arg, getattr(args, arg))
    print(argument_list)

    model = Resnet18_AttEN(class_num=args.class_num)
    model.to(device)
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=1e-5, momentum=0.9, nesterov=True)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)
     
    ds_handler = dataset_handler(args)
    dataset = ds_handler.get_dataset()
    one_batch_dataset = ds_handler.get_dataset(is_one_batch=True)
    cnn_train(model, dataset, args.epochs, optimizer, scheduler, device, tensor_board_path, models_path, args=args)
    cnn_test(model, dataset.test_loader, 'ResNet', device, training_title)