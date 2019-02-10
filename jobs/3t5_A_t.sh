#!/bin/bash
#SBATCH -t 0-05:00:00
#SBATCH -p gpu
#SBATCH -o log/3t5_A_t
cd ..
python test.py --test_window 128 --phase testcube3d --name 3t5_A_1 --dataroot ../../image_standardization_t7505/3t5_3d_small --model pix2pix3d --which_model_netG edsr3_1 --dataset_mode unaligned --input_nc 1 --output_nc 1 --gpu_ids 0 --display_id -1 --serial_batches 2>&1 >> log/3t5_A_t1 
python test.py --test_window 128 --phase testcube3d --name 3t5_A_2 --dataroot ../../image_standardization_t7505/3t5_3d_small --model pix2pix3d --which_model_netG edsr3_2 --dataset_mode unaligned --input_nc 1 --output_nc 1 --gpu_ids 0 --display_id -1 --serial_batches 2>&1 >> log/3t5_A_t2 
python test.py --test_window 128 --phase testcube3d --name 3t5_A_4 --dataroot ../../image_standardization_t7505/3t5_3d_small --model pix2pix3d --which_model_netG edsr3_4 --dataset_mode unaligned --input_nc 1 --output_nc 1 --gpu_ids 0 --display_id -1 --serial_batches 2>&1 >> log/3t5_A_t4
python test.py --test_window 128 --phase testcube3d --name 3t5_A_8 --dataroot ../../image_standardization_t7505/3t5_3d_small --model pix2pix3d --which_model_netG edsr3_8 --dataset_mode unaligned --input_nc 1 --output_nc 1 --gpu_ids 0 --display_id -1 --serial_batches 2>&1 >> log/3t5_A_t8
wait
