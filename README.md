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
python3 [script name] --training_title [folder title] --epochs [num of epoch] --lr [learning rate] --batch_size [batch size] --dataset [isic2019 or fitzpatrick17k] --model [model type] --class_num [8 or 114]  
```

## Evaluation
Using eval_me.py to evaluate the accuracy and fairness scores for model.
```bash
python3 eval.py
```
