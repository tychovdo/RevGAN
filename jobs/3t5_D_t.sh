#!/bin/bash
#SBATCH -t 0-05:00:00
#SBATCH -p gpu
#SBATCH -o log/3t5_D_t
cd ..
python test.py --ngf 32 --ndf 32 --which_epoch 50 --test_window 128 --phase testcube3d --name 3t5_D_1 --dataroot ../../image_standardization_t7505/3t5c_3d --model pix2pix3d --which_model_netG edsr5_1 --dataset_mode unaligned --input_nc 1 --output_nc 1 --gpu_ids 0 --display_id -1 --serial_batches
python test.py --ngf 32 --ndf 32 --which_epoch 50 --test_window 128 --phase testcube3d --name 3t5_D_2 --dataroot ../../image_standardization_t7505/3t5c_3d --model pix2pix3d --which_model_netG edsr5_2 --dataset_mode unaligned --input_nc 1 --output_nc 1 --gpu_ids 0 --display_id -1 --serial_batches
python test.py --ngf 32 --ndf 32 --which_epoch 50 --test_window 128 --phase testcube3d --name 3t5_D_4 --dataroot ../../image_standardization_t7505/3t5c_3d --model pix2pix3d --which_model_netG edsr5_4 --dataset_mode unaligned --input_nc 1 --output_nc 1 --gpu_ids 0 --display_id -1 --serial_batches
wait
