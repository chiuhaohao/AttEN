CUDA_VISIBLE_DEVICES=1 python3 train_resnet18_AttEN.py --training_title 20231130_train_resnet18_isic_AttEN_sp3_ablation_snnl --epochs 200 --lr 0.01 --batch_size 256 --dataset isic2019_mask --model resnet18 --class_num 8 --group_num 2 --image_size 256 --image_crop_size 224 --Lambda 5.0 --sigma 0.5 --alpha 0.01 --prompt True --prompt True --split_num 3

CUDA_VISIBLE_DEVICES=1 python3 train_resnet18_AttEN.py --training_title 20231130_train_resnet18_isic_AttEN_sp2_ablation_snnl --epochs 200 --lr 0.01 --batch_size 256 --dataset isic2019_mask --model resnet18 --class_num 8 --group_num 2 --image_size 256 --image_crop_size 224 --Lambda 5.0 --sigma 0.5 --alpha 0.01 --prompt True --prompt True --split_num 2


# CUDA_VISIBLE_DEVICES=0 python3 train_vgg11_FDKD.py --training_title 20230925_train_vgg11_FDKD_fitz_sp2 --epochs 200 --lr 0.01 --batch_size 64 --dataset fitzpatrick17k_sp2 --model vgg11 --class_num 114 --group_num 2 --image_size 256 --image_crop_size 224 --teacher_path /home/jinghao/Fairness/SAM/networks/121/20230924_train_vgg16_fitz_sp2/195.pth
# CUDA_VISIBLE_DEVICES=0 python3 train_vgg11_FDKD.py --training_title 20230925_train_vgg11_FDKD_fitz_sp3 --epochs 200 --lr 0.01 --batch_size 64 --dataset fitzpatrick17k --model vgg11 --class_num 114 --group_num 2 --image_size 256 --image_crop_size 224 --teacher_path /home/jinghao/Fairness/SAM/networks/121/20230924_train_vgg16_fitz_sp3/195.pth
# CUDA_VISIBLE_DEVICES=0 python3 train_mmd_vgg.py --training_title 20230925_train_vgg11_mfd_isic_sp2 --epochs 200 --lr 0.01 --batch_size 64 --dataset isic2019_sp2 --model vgg11 --class_num 8 --group_num 2 --image_size 256 --image_crop_size 224 --teacher_path /home/jinghao/Fairness/SAM/networks/121/20230924_train_vgg16_isic_sp2/195.pth
# CUDA_VISIBLE_DEVICES=1 python3 train_mmd_vgg.py --training_title 20230925_train_vgg11_mfd_isic_sp3 --epochs 200 --lr 0.01 --batch_size 64 --dataset isic2019 --model vgg11 --class_num 8 --group_num 2 --image_size 256 --image_crop_size 224 --teacher_path /home/jinghao/Fairness/SAM/networks/121/20230924_train_vgg16_isic_sp3/195.pth