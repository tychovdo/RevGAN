#!/bin/bash
#SBATCH -t 0-05:00:00
#SBATCH -p gpu
#SBATCH -o log/3t5_C_4
cd ..
python train.py --ngf 64 --ndf 64 --niter 50 --niter_decay 50 --name 3t5_C_4 --dataroot ../../image_standardization_t7505/3t5c_3d --model pix2pix3d --which_model_netG edsr4_4 --dataset_mode unaligned --input_nc 1 --output_nc 1 --gpu_ids 0 --display_id -1 --serial_batches --batchSize 16 2>&1 >> log/3t5_C_4.log
wait
