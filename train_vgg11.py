import torch
import time
import data
import argparse
import numpy as np
import torch.nn as nn
import torch.optim as optim
import aux_funcs as af

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
            gender  = batch[2].to(device)
            # output, _, _, _, _ = model(b_x)
            output = model(b_x)
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
    
    for epoch in range(1, epochs): 
        CE_loss = [] 
        label_list = []
        y_pred_list = []
        sensitive_group_list = []
        cur_lr = af.get_lr(optimizer)      
        train_loader = data.train_loader
        start_time = time.time()
        model.train()
        print('Epoch: {}/{}'.format(epoch, epochs))
        print('Cur lr: {}'.format(cur_lr))

        for x, y, sensitive_group in tqdm(train_loader):
            b_x = x.to(device)   # batch x
            b_y = y.to(device)   # batch y
            b_sensitive_group = sensitive_group.to(device)
            output = model(b_x)  # cnn final output
            
            _, preds = torch.max(output, 1) 
            
            criterion = af.get_loss_criterion('')
            loss = criterion(output, b_y)
            optimizer.zero_grad()           # clear gradients for this training step
            loss.mean().backward()           # backpropagation, compute gradients
            optimizer.step()   


            CE_loss.append(loss.mean())
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
        print('Epoch took {} seconds.'.format(epoch_time))
        writer.add_scalar('CE Loss: ', sum(CE_loss) / len(CE_loss), epoch)
        writer.add_scalar("Lr/train", cur_lr, epoch)
        print('Start testing...')
        # cnn_test(model, data.test_loader, device)

        if epoch % 5 == 0:
            torch.save(model, '{}/{}.pth'.format(models_path, epoch))
            cnn_test(model, data.test_loader, device)
            
        # if epoch % 5 == 0 and epoch > 99:
        #     torch.save(model, '{}/{}.pth'.format(models_path, epoch))
        #     cnn_test(model, data.test_loader, device)
            

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
    parser.add_argument('--dataset', type=str, default='fitzpatrick17k',
                    help='')
    parser.add_argument('--model', type=str, default='vgg11',
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

    model = models.vgg11(pretrained=True)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, 114)
    # model = models.resnet18(pretrained=True)
    # num_ftrs = model.fc.in_features
    # model.fc = nn.Linear(num_ftrs, 114)
    # /home/jinghao/Fairness/SAM/pruned_weight/20230922_vgg11_fitz_sp2.pth
    # /home/jinghao/Fairness/SAM/pruned_weight/20230922_vgg11_fitz_sp3.pth
    # /home/jinghao/Fairness/SAM/pruned_weight/20230922_vgg11_isic_sp2.pth
    # /home/jinghao/Fairness/SAM/pruned_weight/20230922_vgg11_isic_sp3.pth
    # model = torch.load('/home/jinghao/Fairness/SAM/pruned_weight/20230922_vgg11_isic_sp2.pth')
    # for i in range(19):
    #     idx = 100 + i*5
    # load_path = '/home/jinghao/Fairness/SAM/pruned_weight/resnet18_fitz_split3.pth'
    # checkpoint = torch.load(load_path)
    # model.load_state_dict(checkpoint['model_dict'])
    # model = torch.load(load_path)
    model = torch.load('/home/jinghao/Fairness/SAM/networks/121/20240411_vgg11_fitz_sp3/100.pth')
    model.to(device)
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=1e-5, momentum=0.9, nesterov=True)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)

    
    ds_handler = dataset_handler(args)
    dataset = ds_handler.get_dataset()
    one_batch_dataset = ds_handler.get_dataset(is_one_batch=True)
    

    cnn_train(model, dataset, args.epochs, optimizer, scheduler, device, tensor_board_path, models_path, args=args)
    # cnn_test(model, one_batch_dataset.vali_loader, device)
    # cnn_test(model, one_batch_dataset.test_loader, device)