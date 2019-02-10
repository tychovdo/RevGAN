#!/bin/bash
#SBATCH -t 0-05:00:00
#SBATCH -p gpu
#SBATCH -o log/3t5_B_t
cd ..
python test.py --which_epoch 50 --test_window 128 --phase testcube3d --name 3t5_B_1 --dataroot ../../image_standardization_t7505/3t5b_3d --model pix2pix3d --which_model_netG edsr3_1 --dataset_mode unaligned --input_nc 1 --output_nc 1 --gpu_ids 0 --display_id -1 --serial_batches
python test.py --which_epoch 50 --test_window 128 --phase testcube3d --name 3t5_B_2 --dataroot ../../image_standardization_t7505/3t5b_3d --model pix2pix3d --which_model_netG edsr3_2 --dataset_mode unaligned --input_nc 1 --output_nc 1 --gpu_ids 0 --display_id -1 --serial_batches
python test.py --which_epoch 50 --test_window 128 --phase testcube3d --name 3t5_B_4 --dataroot ../../image_standardization_t7505/3t5b_3d --model pix2pix3d --which_model_netG edsr3_4 --dataset_mode unaligned --input_nc 1 --output_nc 1 --gpu_ids 0 --display_id -1 --serial_batches
python test.py --which_epoch 50 --test_window 128 --phase testcube3d --name 3t5_B_8 --dataroot ../../image_standardization_t7505/3t5b_3d --model pix2pix3d --which_model_netG edsr3_8 --dataset_mode unaligned --input_nc 1 --output_nc 1 --gpu_ids 0 --display_id -1 --serial_batches
wait
