#!/bin/bash
#SBATCH -t 0-05:00:00
#SBATCH -p gpu
#SBATCH -o log/1024_C_t
cd ..
python test.py --which_epoch 25 --test_window 128 --phase testcube3d --name 1024_C_1 --dataroot ../../image_standardization_t7505/1024b_3d --model pix2pix3d --which_model_netG edsr4_1 --dataset_mode unaligned --input_nc 1 --output_nc 1 --gpu_ids 0 --display_id -1 --serial_batches
python test.py --which_epoch 25 --test_window 128 --phase testcube3d --name 1024_C_2 --dataroot ../../image_standardization_t7505/1024b_3d --model pix2pix3d --which_model_netG edsr4_2 --dataset_mode unaligned --input_nc 1 --output_nc 1 --gpu_ids 0 --display_id -1 --serial_batches 
python test.py --which_epoch 25 --test_window 128 --phase testcube3d --name 1024_C_4 --dataroot ../../image_standardization_t7505/1024b_3d --model pix2pix3d --which_model_netG edsr4_4 --dataset_mode unaligned --input_nc 1 --output_nc 1 --gpu_ids 0 --display_id -1 --serial_batches
wait
