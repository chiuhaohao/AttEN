# Achieve fairness without demographics for dermatological disease diagnosis [Medical Image Analysis]

This is the official repository of the following paper:

**Achieve fairness without demographics for dermatological disease diagnosis**<br>
Ching-Hao Chiu, Yu-Jen Chen, Yawen Wu, Yiyu Shi, Tsung-Yi Ho

[[Paper](https://www.sciencedirect.com/science/article/abs/pii/S1361841524001130)] 


## Setup & Preparation
### Environment setup
```bash
pip install -r requirements.txt
```

## Training
For training, using the command in train.sh to train the AttEN model.
The command is as follow
```bash
CUDA_VISIBLE_DEVICES=0 python3 [script name] --training_title [title name] --epochs [num of epoch] --lr [learning rate] --batch_size [batch size] --dataset [isic2019_mask or fitzpatrick17k_mask] --model [model type] --class_num [8 or 114] --group_num 2 --image_size 256 --image_crop_size 224 --Lambda 5.0 --sigma 0.5 --alpha 0.01 --prompt True --split_num [data split num]
```

## Evaluation
The training code will evaluate the model automatically.
