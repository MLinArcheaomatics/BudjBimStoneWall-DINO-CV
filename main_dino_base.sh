#!/bin/bash
#SBATCH --mem=64gb
#SBATCH --output=./output/%j_log.out

module load Anaconda3/2024.02-1
eval "$(conda shell.bash hook)"
conda activate torch

 python3 main_dino_cv_run_with_submitit.py \
 --nodes 1 \
 --ngpus 2 \
 --mem_gb 32 \
 --partition feit-gpu-a100 \
 --qos feit \
 --time 2-0:0:0 \
 --job_name main_dino_cv \
 --arch resnet50 \
 --use_bn_in_head True \
 --use_fp16 False \
 --lr 0.001 \
 --optimizer sgd \
 --global_crops_scale 0.15 1.0 \
 --min_scale_crops 0.05 \
 --max_scale_crops 0.15 \
 --data_view mhs \
 --data_path ./data/budjbim_landscape \
 --pretrained \
 --pretrained_model_ckpt ./save/pretrained_weights/dino_resnet50_pretrain.pth \
 --output_dir ./save/pretrained_weights/dino_base/rn50_mhs_budjbim_pretrained \
 --num_workers 8

 python3 main_dino_cv_run_with_submitit.py \
 --nodes 1 \
 --ngpus 2 \
 --mem_gb 32 \
 --partition feit-gpu-a100 \
 --qos feit \
 --time 2-0:0:0 \
 --job_name main_dino_cv \
 --arch resnet50 \
 --use_bn_in_head True \
 --use_fp16 False \
 --lr 0.001 \
 --optimizer sgd \
 --global_crops_scale 0.15 1.0 \
 --min_scale_crops 0.05 \
 --max_scale_crops 0.15 \
 --data_view vat \
 --data_path ./data/budjbim_landscape \
 --pretrained \
 --pretrained_model_ckpt ./save/pretrained_weights/dino_resnet50_pretrain.pth \
 --output_dir ./save/pretrained_weights/dino_base/rn50_vat_budjbim_pretrained \
 --num_workers 8
 
 python3 main_dino_cv_run_with_submitit.py \
 --nodes 1 \
 --ngpus 2 \
 --mem_gb 32 \
 --partition feit-gpu-a100 \
 --qos feit \
 --time 2-0:0:0 \
 --job_name main_dino_cv \
 --arch wide_resnet50_2 \
 --use_bn_in_head True \
 --use_fp16 False \
 --lr 0.001 \
 --optimizer sgd \
 --global_crops_scale 0.15 1.0 \
 --min_scale_crops 0.05 \
 --max_scale_crops 0.15 \
 --data_view mhs \
 --data_path ./data/budjbim_landscape \
 --pretrained \
 --pretrained_model_ckpt ./save/pretrained_weights/dino_mc_wide_resnet50.pth \
 --output_dir ./save/pretrained_weights/dino_base/wrn50_mhs_budjbim_pretrained \
 --num_workers 8

 python3 main_dino_cv_run_with_submitit.py \
 --nodes 1 \
 --ngpus 2 \
 --mem_gb 32 \
 --partition feit-gpu-a100 \
 --qos feit \
 --time 2-0:0:0 \
 --job_name main_dino_cv \
 --arch wide_resnet50_2 \
 --use_bn_in_head True \
 --use_fp16 False \
 --lr 0.001 \
 --optimizer sgd \
 --global_crops_scale 0.15 1.0 \
 --min_scale_crops 0.05 \
 --max_scale_crops 0.15 \
 --data_view vat \
 --data_path ./data/budjbim_landscape \
 --pretrained \
 --pretrained_model_ckpt ./save/pretrained_weights/dino_mc_wide_resnet50.pth \
 --output_dir ./save/pretrained_weights/dino_base/wrn50_vat_budjbim_pretrained \
 --num_workers 8

 python3 main_dino_cv_run_with_submitit.py \
 --nodes 1 \
 --ngpus 2 \
 --mem_gb 32 \
 --partition feit-gpu-a100 \
 --qos feit \
 --time 2-0:0:0 \
 --job_name main_dino_cv \
 --arch vit_small \
 --patch_size 16 \
 --norm_last_layer False \
 --use_bn_in_head False \
 --use_fp16 False \
 --optimizer adamw \
 --data_view mhs \
 --data_path ./data/budjbim_landscape \
 --pretrained \
 --pretrained_model_ckpt ./save/pretrained_weights/dino_deitsmall16_pretrain.pth \
 --output_dir ./save/pretrained_weights/dino_base/vitsmall16_mhs_budjbim_pretrained \
 --num_workers 8 

 python3 main_dino_cv_run_with_submitit.py \
 --nodes 1 \
 --ngpus 2 \
 --mem_gb 32 \
 --partition feit-gpu-a100 \
 --qos feit \
 --time 2-0:0:0 \
 --job_name main_dino_cv \
 --arch vit_small \
 --patch_size 16 \
 --norm_last_layer False \
 --use_bn_in_head False \
 --use_fp16 False \
 --optimizer adamw \
 --data_view vat \
 --data_path ./data/budjbim_landscape \
 --pretrained \
 --pretrained_model_ckpt ./save/pretrained_weights/dino_deitsmall16_pretrain.pth \
 --output_dir ./save/pretrained_weights/dino_base/vitsmall16_vat_budjbim_pretrained \
 --num_workers 8 

 