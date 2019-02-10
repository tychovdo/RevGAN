import numpy as np
from PIL import Image
import os

import argparse 
from sklearn.metrics import mean_absolute_error as compare_mae
from skimage.measure import compare_psnr
from skimage.measure import compare_ssim

parser = argparse.ArgumentParser()
parser.add_argument("--result_dir", type=str, required=True, help="Path to the generated images to be evaluated")
parser.add_argument("--version", type=str, default='testcube_latest', help="Version of model")
args = parser.parse_args()

path = os.path.join(args.result_dir, args.version)
output_file = os.path.join(args.result_dir, args.version, 'scores.txt')

with open(output_file, 'w') as f:
    # Load data
    fake_Bs = sorted([os.path.join(path, x) for x in os.listdir(path) if 'fake_B' in x])
    real_Bs = sorted([os.path.join(path, x) for x in os.listdir(path) if 'real_B' in x])
    fake_As = sorted([os.path.join(path, x) for x in os.listdir(path) if 'fake_A' in x])

    is_backward = True if len(fake_As) > 0 else False
    if is_backward:
        real_As = sorted([os.path.join(path, x) for x in os.listdir(path) if 'real_A' in x])

    f.write('Model: {}\tVersion: {}\tFwrd: {}  \tBwrd: {}'.format(args.result_dir, args.version, len(fake_Bs), len(fake_As)))

    # IDENTITY BASELINE

    # Forward
    mae_s, psnr_s, ssim_s = [], [], []
    for real_A_path, real_B_path in zip(real_As, real_Bs):
        im_pred = np.load(real_A_path) * 1000
        im_true = np.load(real_B_path) * 1000
        mae_s.append(compare_mae(im_true.flatten(), im_pred.flatten()))
        psnr_s.append(compare_psnr(im_true, im_pred, data_range=4096))
        ssim_s.append(compare_ssim(im_true, im_pred, data_range=4096))

    f.write('A->B')
    f.write('MAE:  %s_%s' % (np.mean(mae_s), np.std(mae_s)))
    f.write('PSNR: %s_%s' % (np.mean(psnr_s), np.std(psnr_s)))
    f.write('SSIM: %s_%s' % (np.mean(ssim_s), np.std(ssim_s)))

    # Backward
    mae_s, psnr_s, ssim_s = [], [], []
    for real_B_path, real_A_path in zip(real_Bs, real_As):
        im_pred = np.load(real_B_path) * 1000
        im_true = np.load(real_A_path) * 1000
        mae_s.append(compare_mae(im_true.flatten(), im_pred.flatten()))
        psnr_s.append(compare_psnr(im_true, im_pred, data_range=4096))
        ssim_s.append(compare_ssim(im_true, im_pred, data_range=4096))

    f.write('B->A')
    f.write('MAE:  %s_%s' % (np.mean(mae_s), np.std(mae_s)))
    f.write('PSNR: %s_%s' % (np.mean(psnr_s), np.std(psnr_s)))
    f.write('SSIM: %s_%s' % (np.mean(ssim_s), np.std(ssim_s)))


    # REAL RESTULS

    # Forward
    mae_s, psnr_s, ssim_s = [], [], []
    for fake_B_path, real_B_path in zip(fake_Bs, real_Bs):
        im_pred = np.load(fake_B_path) * 1000
        im_true = np.load(real_B_path) * 1000
        mae_s.append(compare_mae(im_true.flatten(), im_pred.flatten()))

        psnr_s.append(compare_psnr(im_true, im_pred, data_range=4096))
        ssim_s.append(compare_ssim(im_true, im_pred, data_range=4096))

    f.write('A->B')
    f.write('MAE:  %s_%s' % (np.mean(mae_s), np.std(mae_s)))
    f.write('PSNR: %s_%s' % (np.mean(psnr_s), np.std(psnr_s)))
    f.write('SSIM: %s_%s' % (np.mean(ssim_s), np.std(ssim_s)))

    # Backward
    if is_backward:
        mae_s, psnr_s, ssim_s = [], [], []
        for fake_A_path, real_A_path in zip(fake_As, real_As):
            im_pred = np.load(fake_A_path) * 1000
            im_true = np.load(real_A_path) * 1000
            mae_s.append(compare_mae(im_true.flatten(), im_pred.flatten()))
            psnr_s.append(compare_psnr(im_true, im_pred, data_range=4096))
            ssim_s.append(compare_ssim(im_true, im_pred, data_range=4096))

        f.write('B->A')
        f.write('MAE, PSNR, SSIM')
        f.write('MAE:  %s_%s' % (np.mean(mae_s), np.std(mae_s)))
        f.write('PSNR: %s_%s' % (np.mean(psnr_s), np.std(psnr_s)))
        f.write('SSIM: %s_%s' % (np.mean(ssim_s), np.std(ssim_s)))
