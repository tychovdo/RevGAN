#!/bin/bash
#SBATCH -t 0-05:00:00
#SBATCH -p gpu
#SBATCH -o log/3t5b_edsr3_1
cd ..
python train.py --name 3t5b_edsr3_1 --dataroot ../../image_standardization_t7505/3t5_3d --model pix2pix3d --which_model_netG edsr3_1 --dataset_mode unaligned --input_nc 1 --output_nc 1 --gpu_ids 0 --display_id -1 --serial_batches --display_freq 1 --update_html_freq 1 --print_freq 1 2>&1 >> log/3t5b_edsr3_1.log 
wait
