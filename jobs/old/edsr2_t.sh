#!/bin/bash
#SBATCH -t 0-05:00:00
#SBATCH -p gpu
#SBATCH -o log/edsr2_t
cd ..
python test.py --phase testcube3d --name edsr2_1 --dataroot ../../image_standardization_t7505/1024_3d_small --model pix2pix3d --which_model_netG edsr2_1 --dataset_mode unaligned --input_nc 1 --output_nc 1 --gpu_ids -1 --display_id -1 --serial_batches 2>&1 >> log/edsr2_t1 
python test.py --phase testcube3d --name edsr2_2 --dataroot ../../image_standardization_t7505/1024_3d_small --model pix2pix3d --which_model_netG edsr2_2 --dataset_mode unaligned --input_nc 1 --output_nc 1 --gpu_ids -1 --display_id -1 --serial_batches 2>&1 >> log/edsr2_t2 
python test.py --phase testcube3d --name edsr2_4 --dataroot ../../image_standardization_t7505/1024_3d_small --model pix2pix3d --which_model_netG edsr2_4 --dataset_mode unaligned --input_nc 1 --output_nc 1 --gpu_ids -1 --display_id -1 --serial_batches 2>&1 >> log/edsr2_t4
python test.py --phase testcube3d --name edsr2_8 --dataroot ../../image_standardization_t7505/1024_3d_small --model pix2pix3d --which_model_netG edsr2_8 --dataset_mode unaligned --input_nc 1 --output_nc 1 --gpu_ids -1 --display_id -1 --serial_batches 2>&1 >> log/edsr2_t8
wait
