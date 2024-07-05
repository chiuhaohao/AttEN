from torch.utils.data import Dataset
from data_augmentation import *
import torch
import os 
import io
import skimage
from skimage import io
from torch.utils.data.sampler import WeightedRandomSampler
import numpy as np
import pandas as pd

class dataset_handler:
    def __init__(self, args):
        self.args = args
        self.num_class = {
            'isic2019_mask' : 8,
            'fitzpatrick17k_mask' : 114
        }
        
    def get_num_class(self):
        return self.num_class[self.args.dataset]
    
    def get_dataset(self, is_one_batch=False):
        if is_one_batch:
            batch_size = 1
        else:
            batch_size = self.args.batch_size
    
        dataset = self.get_dataset_class()(batch_size=batch_size, sp=self.args.split_num)

        return dataset
    
    def get_dataset_class(self):
        if self.args.dataset == 'isic2019_mask':
            return ISIC2019_mask
        if self.args.dataset == 'fitzpatrick17k_mask':
            return fitzpatrick17k_mask

def get_weighted_sampler(df, label_level = 'low'):
    class_sample_count = np.array(df[label_level].value_counts().sort_index())
    class_weight = 1. / class_sample_count
    # print('count: ', class_sample_count)
    result = []
    for t in df[label_level]:
        if t == 4:
            t = 2
        result.append(class_weight[t])
    samples_weight = np.array(result)
    samples_weight = torch.from_numpy(samples_weight)
    sampler = WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'), len(samples_weight), replacement=True)
    return sampler

def ISIC2019_holdout_gender(df, holdout_set: str = 'none'):
    if holdout_set == "0":
        remain_df = df[df.gender==1].reset_index(drop=True)
    elif holdout_set == "1":
        remain_df = df[df.gender==0].reset_index(drop=True)
    else:
        remain_df = df
    return remain_df


class ISIC2019_mask_dataset_transform(Dataset):

    def __init__(self, df=None, root_dir=None, mask_root_dir=None, transform=True, feature_dict=None):
        assert df is not None
        self.df = df
        self.root_dir = root_dir
        self.mask_root_dir = mask_root_dir
        self.transform = transform
        self.feature_dict = feature_dict

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_name = os.path.join(self.root_dir, self.df.loc[self.df.index[idx], 'image']+'.jpg')
        mask_img_name = os.path.join(self.mask_root_dir, self.df.loc[self.df.index[idx], 'image']+'_mask.jpg')
        image = io.imread(img_name)
        mask_image = io.imread(mask_img_name)
        # some images have alpha channel, we just not ignore alpha channel
        if (image.shape[0] > 3):
            image = image[:,:,:3]
        if(len(image.shape) < 3):
            image = skimage.color.gray2rgb(image)
        if self.transform:
            image = self.transform(image)
        
        # print('image shape:', image.shape)
        if (mask_image.shape[0] > 3):
            mask_image = mask_image[:,:,:3]
        if(len(mask_image.shape) < 3):
            mask_image = skimage.color.gray2rgb(mask_image)
        if self.transform:
            mask_image = self.transform(mask_image)
        # print('mask image:', mask_image)
            
        label = self.df.loc[self.df.index[idx], 'low']
        gender = self.df.loc[self.df.index[idx], 'gender']
        feature = {}
        if self.feature_dict != None:
            feature[0] = torch.Tensor(self.feature_dict[0][label])
            feature[1] = torch.Tensor(self.feature_dict[1][label])
            feature[2] = torch.Tensor(self.feature_dict[2][label])
            feature[3] = torch.Tensor(self.feature_dict[3][label])
            feature[4] = torch.squeeze(torch.Tensor(self.feature_dict[4][label]))
            return image, label, mask_image, gender, feature
        else:
            return image, label, mask_image, gender

def ISIC2019_holdout_gender(df, holdout_set: str = 'none'):
    if holdout_set == "0":
        remain_df = df[df.gender==1].reset_index(drop=True)
    elif holdout_set == "1":
        remain_df = df[df.gender==0].reset_index(drop=True)
    else:
        remain_df = df
    return remain_df

class ISIC2019_mask:
    def __init__(self, batch_size=64, add_trigger=False, model_name=None, feature_dict=None, sp=1):
        self.batch_size = batch_size
        self.num_classes = 9
        if model_name == 'ResNet':
            self.image_size = 128
        else:
            self.image_size = 224 # for VGG

        predefined_root_dir = '/ISIC_2019_Training_Input' # specify the image dir
        predefined_mask_root_dir = '/ISIC_2019_Training_Input_mask'
        train_df = pd.read_csv('/isic2019_split/isic2019_train_pretraining.csv')
        vali_df = pd.read_csv('/isic2019_split/isic2019_val_pretraining.csv')
        test_df = pd.read_csv('/isic2019_split/isic2019_test_pretraining.csv')
        if(sp == 1):
            print('--------------------sp1---------------------------')
            train_df = pd.read_csv('/isic2019_split/isic2019_train_pretraining.csv')
            vali_df = pd.read_csv('/isic2019_split/isic2019_val_pretraining.csv')
            test_df = pd.read_csv('/isic2019_split/isic2019_test_pretraining.csv')
        elif(sp == 2):
            print('--------------------sp2---------------------------')
            train_df = pd.read_csv('/isic2019_split/isic2019_train_split2.csv')
            vali_df = pd.read_csv('/isic2019_split/isic2019_val_split2.csv')
            test_df = pd.read_csv('/isic2019_split/isic2019_test_split2.csv')
        elif(sp == 3):
            print('--------------------sp3---------------------------')
            train_df = pd.read_csv('/isic2019_split/isic2019_train_split3.csv')
            vali_df = pd.read_csv('/isic2019_split/isic2019_val_split3.csv')
            test_df = pd.read_csv('/isic2019_split/isic2019_test_split3.csv')
        use_cuda = torch.cuda.is_available()
        
        kwargs = {'num_workers': 16, 'pin_memory': True} if use_cuda else {}
        sampler = get_weighted_sampler(train_df, label_level='low')
        train_transform = ISIC2019_Augmentations(is_training=True, image_size=256, input_size=224, model_name=model_name).transforms
        test_transform = ISIC2019_Augmentations(is_training=False, image_size=256, input_size=224, model_name=model_name).transforms
        aug_trainset =  ISIC2019_mask_dataset_transform(df=train_df, root_dir=predefined_root_dir, mask_root_dir=predefined_mask_root_dir, transform=train_transform, feature_dict=feature_dict)
        self.aug_train_loader = torch.utils.data.DataLoader(aug_trainset, batch_size=self.batch_size, sampler=sampler, **kwargs)
        train_dataset = ISIC2019_mask_dataset_transform(df=train_df, root_dir=predefined_root_dir, mask_root_dir=predefined_mask_root_dir, transform=train_transform, feature_dict=feature_dict)
        self.train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, sampler=sampler, **kwargs)
        vali_dataset= ISIC2019_mask_dataset_transform(df=vali_df, root_dir=predefined_root_dir, mask_root_dir=predefined_mask_root_dir, transform=test_transform, feature_dict=feature_dict)
        self.vali_loader = torch.utils.data.DataLoader(vali_dataset, batch_size=batch_size, shuffle=False, **kwargs)
        test_dataset= ISIC2019_mask_dataset_transform(df=test_df, root_dir=predefined_root_dir, mask_root_dir=predefined_mask_root_dir, transform=test_transform, feature_dict=feature_dict)
        self.test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, **kwargs)

        if add_trigger:  #skip this
            self.trigger_test_set = None
            self.trigger_test_loader = None

class fitzpatrick17k_mask:
    def __init__(self, batch_size=64, add_trigger=False, model_name=None, sp=1):
        self.batch_size = batch_size
        self.num_classes = 114

        predefined_root_dir = '/fitzpatrick17k_dataset_images' # specify the image dir
        predefined_mask_root_dir = '/fitz_mask'
        if(sp == 1):
            print('---------sp1_fitz-------------')
            train_df = pd.read_csv('/fitzpatrick17k/fitzpatrick_train_split1.csv')
            vali_df = pd.read_csv('/fitzpatrick17k/fitzpatrick_val_split1.csv')
            test_df = pd.read_csv('/fitzpatrick17k/fitzpatrick_test_split1.csv')
        if(sp == 2):
            print('---------sp2_fitz-------------')
            train_df = pd.read_csv('/fitzpatrick17k/fitzpatrick_train_split2.csv')
            vali_df = pd.read_csv('/fitzpatrick17k/fitzpatrick_val_split2.csv')
            test_df = pd.read_csv('/fitzpatrick17k/fitzpatrick_test_split2.csv')
        if(sp == 3):
            print('---------sp3_fitz-------------')
            train_df = pd.read_csv('/fitzpatrick17k/fitzpatrick_train_split3.csv')
            vali_df = pd.read_csv('/fitzpatrick17k/fitzpatrick_val_split3.csv')
            test_df = pd.read_csv('/fitzpatrick17k/fitzpatrick_test_split3.csv')
        use_cuda = torch.cuda.is_available()
        kwargs = {'num_workers': 16, 'pin_memory': True} if use_cuda else {}
        sampler = get_weighted_sampler(train_df, label_level='low')

        train_transform = ISIC2019_Augmentations(is_training=True, image_size=256, input_size=224, model_name=model_name).transforms
        test_transform = ISIC2019_Augmentations(is_training=False, image_size=256, input_size=224, model_name=model_name).transforms
        aug_trainset =  Fitzpatrick17k_mask_dataset_transform(df=train_df, root_dir=predefined_root_dir, mask_root_dir=predefined_mask_root_dir, transform=train_transform)
        self.aug_train_loader = torch.utils.data.DataLoader(aug_trainset, batch_size=self.batch_size, sampler=sampler, **kwargs)
        train_dataset = Fitzpatrick17k_mask_dataset_transform(df=train_df, root_dir=predefined_root_dir, mask_root_dir=predefined_mask_root_dir, transform=train_transform)
        self.train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, sampler=sampler, **kwargs)
        vali_dataset= Fitzpatrick17k_mask_dataset_transform(df=vali_df, root_dir=predefined_root_dir, mask_root_dir=predefined_mask_root_dir, transform=test_transform)
        self.vali_loader = torch.utils.data.DataLoader(vali_dataset, batch_size=batch_size, shuffle=False, **kwargs)
        test_dataset= Fitzpatrick17k_mask_dataset_transform(df=test_df, root_dir=predefined_root_dir, mask_root_dir=predefined_mask_root_dir, transform=test_transform)
        self.test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, **kwargs)

        if add_trigger:  #skip this
            self.trigger_test_set = None
            self.trigger_test_loader = None

class Fitzpatrick17k_mask_dataset_transform(Dataset):

    def __init__(self, df=None, root_dir=None, mask_root_dir=None, transform=None):
        """
        Args:
            train: True for training, False for testing
            transform (callable, optional): Optional transform to be applied
                on a sample.
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
        """
        assert df is not None
        self.df = df
        self.root_dir = root_dir
        self.mask_root_dir = mask_root_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_name = os.path.join(self.root_dir,
                                self.df.loc[self.df.index[idx], 'hasher']+'.jpg')
        mask_img_name = os.path.join(self.mask_root_dir, self.df.loc[self.df.index[idx], 'hasher']+'_mask.jpg')
        image = io.imread(img_name)
        mask_image = io.imread(mask_img_name)
        if(len(image.shape) < 3):
            image = skimage.color.gray2rgb(image)
        if(len(mask_image.shape) < 3):
            mask_image = skimage.color.gray2rgb(mask_image)

        hasher = self.df.loc[self.df.index[idx], 'hasher']
        high = self.df.loc[self.df.index[idx], 'high']
        mid = self.df.loc[self.df.index[idx], 'mid']
        low = self.df.loc[self.df.index[idx], 'low']
        fitzpatrick = self.df.loc[self.df.index[idx], 'fitzpatrick']
        if 1 <= fitzpatrick <= 3:
            skin_color_binary = 0
        elif 4 <= fitzpatrick <= 6:
            skin_color_binary = 1
        # if 1 <= fitzpatrick <= 6:
        #     skin_color_binary = fitzpatrick
        else:
            skin_color_binary = -1
        if self.transform:
            image = self.transform(image)
        if self.transform:
            mask_image = self.transform(mask_image)
        label = self.df.loc[self.df.index[idx], 'low']
        sample = {
                    'image': image,
                    'high': high,
                    'mid': mid,
                    'low': low,
                    'hasher': hasher,
                    'fitzpatrick': fitzpatrick,
                    'skin_color_binary': skin_color_binary,
                }
        return image, label, mask_image, skin_color_binary