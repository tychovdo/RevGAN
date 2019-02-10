#!/bin/bash
#SBATCH -t 0-05:00:00
#SBATCH -p gpu
#SBATCH -o log/3t5_Dl_2
cd ..
python train.py --ngf 32 --ndf 32 --niter 50 --niter_decay 50 --name 3t5_D_2 --dataroot ../../image_standardization_t7505/3t5c_3d --model pix2pix3d --which_model_netG edsr5_2 --dataset_mode unaligned --input_nc 1 --output_nc 1 --gpu_ids 0 --display_id -1 --serial_batches 2>&1 >> log/3t5_D_2.log
wait
