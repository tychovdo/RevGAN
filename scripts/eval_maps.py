import numpy as np
from PIL import Image
import os, sys

import argparse 
from sklearn.metrics import mean_absolute_error as compare_mae
from skimage.measure import compare_psnr
from skimage.measure import compare_ssim

from labels import labels
import scipy, skimage
from scipy.spatial import KDTree
from sklearn.metrics import confusion_matrix
import caffe
from util import *


 
parser = argparse.ArgumentParser()
parser.add_argument("--result_dir", type=str, required=True, help="Path to the generated images to be evaluated")
parser.add_argument("--version", type=str, default='test_latest', help="Version of model")
parser.add_argument("--which_direction", type=str, default='AtoB', help="direction")
parser.add_argument("--caffemodel_dir", type=str, default='./scripts/caffemodel/', help="Where the FCN-8s caffemodel stored")
args = parser.parse_args()

# Set paths
img_path = os.path.join(args.result_dir, args.version, 'images')
output_file = os.path.join(args.result_dir, args.version, 'scores.txt')


with open(output_file, 'w') as f:
    # Load data
    real_As = sorted([os.path.join(img_path, x) for x in os.listdir(img_path) if 'real_A' in x])
    fake_Bs = sorted([os.path.join(img_path, x) for x in os.listdir(img_path) if 'fake_B' in x])
    real_Bs = sorted([os.path.join(img_path, x) for x in os.listdir(img_path) if 'real_B' in x])
    fake_As = sorted([os.path.join(img_path, x) for x in os.listdir(img_path) if 'fake_A' in x])
    
    is_forward = True
    is_backward = True if len(fake_As) > 0 else False
    
    f.write('Model: {}\tVersion: {}\tFwrd: {}  \tBwrd: {}\n'.format(args.result_dir, args.version, len(fake_Bs), len(fake_As)))
    
    # Direction
    if args.which_direction == 'BtoA':
        is_forward, is_backward = is_backward, is_forward
        real_As, real_Bs, fake_As, fake_Bs = real_Bs, real_As, fake_Bs, fake_As
    
    # Forward
    if is_forward:
        print('forward...')
        mae, psnr, ssim = [], [], []
        for i, (fakeb_path, realb_path) in enumerate(zip(fake_Bs, real_Bs)):
            print(i, len(fake_Bs))
            fakeb = np.array(Image.open(fakeb_path))
            realb = np.array(Image.open(realb_path))

            mae.append(compare_mae(realb.flatten(), fakeb.flatten()))
            psnr.append(compare_psnr(realb.flatten(), fakeb.flatten()))
            ssim.append(compare_ssim(realb.flatten(), fakeb.flatten()))
        mae = np.array(mae)
        psnr = np.array(psnr)
        ssim = np.array(ssim)

        f.write('\nBackward Scores\n')
        f.write('MAE:  %s %s\n' % (mae.mean(), mae.std()))
        f.write('PSNR: %s %s\n' % (psnr.mean(), psnr.std()))
        f.write('SSIM: %s %s\n' % (ssim.mean(), ssim.std()))
    
    # Backward
    if is_backward:
        print('backward...')
        mae, psnr, ssim = [], [], []
        for i, (fakea_path, reala_path) in enumerate(zip(fake_As, real_As)):
            print(i, len(fake_As))
            
            fakea = np.array(Image.open(fakea_path))
            reala = np.array(Image.open(reala_path))

            mae.append(compare_mae(reala.flatten(), fakea.flatten()))
            psnr.append(compare_psnr(reala.flatten(), fakea.flatten()))
            ssim.append(compare_ssim(reala.flatten(), fakea.flatten()))

        mae = np.array(mae)
        psnr = np.array(psnr)
        ssim = np.array(ssim)

        f.write('\nBackward Scores\n')
        f.write('MAE:  %s %s\n' % (mae.mean(), mae.std()))
        f.write('PSNR: %s %s\n' % (psnr.mean(), psnr.std()))
        f.write('SSIM: %s %s\n' % (ssim.mean(), ssim.std()))

