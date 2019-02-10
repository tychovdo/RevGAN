#!/bin/bash
#SBATCH -t 0-05:00:00
#SBATCH -p gpu
#SBATCH -o log/edsr_t
cd ..
python test.py --name edsr1 --dataroot ../../image_standardization_t7505/1024_3d_small --model paired_revgan3d --which_model_netG edsr_1 --dataset_mode unaligned --input_nc 1 --output_nc 1 --gpu_ids 0 --display_id -1 --serial_batches --phase testcube3d 2>&1 >> log/edsr_t1 
python test.py --name edsr2 --dataroot ../../image_standardization_t7505/1024_3d_small --model paired_revgan3d --which_model_netG edsr_2 --dataset_mode unaligned --input_nc 1 --output_nc 1 --gpu_ids 0 --display_id -1 --serial_batches --phase testcube3d 2>&1 >> log/edsr_t2
python test.py --name edsr4 --dataroot ../../image_standardization_t7505/1024_3d_small --model paired_revgan3d --which_model_netG edsr_4 --dataset_mode unaligned --input_nc 1 --output_nc 1 --gpu_ids 0 --display_id -1 --serial_batches --phase testcube3d 2>&1 >> log/edsr_t4 
python test.py --name edsr8 --dataroot ../../image_standardization_t7505/1024_3d_small --model paired_revgan3d --which_model_netG edsr_8 --dataset_mode unaligned --input_nc 1 --output_nc 1 --gpu_ids 0 --display_id -1 --serial_batches --phase testcube3d 2>&1 >> log/edsr_t8 
wait
