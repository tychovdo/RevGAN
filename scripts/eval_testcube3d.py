import numpy as np
from PIL import Image
import os
from skimage.transform import resize

from scipy.ndimage.filters import convolve
import argparse 
from sklearn.metrics import mean_squared_error as compare_mse
from sklearn.metrics import mean_absolute_error as compare_mae
from skimage.measure import compare_psnr
from skimage.measure import compare_ssim

compare_mae5 = lambda a, b: compare_mae(a, b) * 1e5
compare_rmse = lambda a, b: np.sqrt(compare_mse(a, b))
compare_rmse5 = lambda a, b: np.sqrt(compare_mse(a, b)) * 1e5

resize2 = lambda x, y: resize(x, output_shape=y, preserve_range=True, anti_aliasing=True, mode='constant')

parser = argparse.ArgumentParser()
parser.add_argument("--result_dir", type=str, required=True, help="Path to the generated images to be evaluated")
parser.add_argument("--version", type=str, default='testcube3d_latest', help="Version of model")
args = parser.parse_args()

path = os.path.join(args.result_dir, args.version)
output_file = os.path.join(args.result_dir, args.version, 'scores.txt')



with open(output_file, 'w') as f:
    print('Loading data paths...')
    # Load data
    real_As = sorted([os.path.join(path, x) for x in os.listdir(path) if 'real_A' in x])
    fake_Bs = sorted([os.path.join(path, x) for x in os.listdir(path) if 'fake_B' in x])
    real_Bs = sorted([os.path.join(path, x) for x in os.listdir(path) if 'real_B' in x])

    f.write('Model: {}\tVersion: {}\tFwrd: {}  \n'.format(args.result_dir, args.version, len(fake_Bs)))

    # Forward
    rmse_s, mae_s, psnr_s, ssim_s = [], [], [], []
    # rmse_bil_s, mae_bil_s, psnr_bil_s, ssim_bil_s = [], [], [], []
    for fake_B_path, real_B_path in zip(fake_Bs, real_Bs):
        print('Loading real_B:', real_B_path)
        im_true = np.load(real_B_path).flatten()
        if False:
            print(im_true.shape)
            im_bil = resize2(resize2(im_true, (im_true.shape[0]//4, im_true.shape[1]//2, im_true.shape[2]//2)), im_true.shape).flatten()
            print('comparing trilinear...')
            rmse = compare_rmse(im_true, im_bil)
            rmse_bil_s.append(rmse)
            print('rmse: ', rmse)
            mae = compare_mae(im_true, im_bil)
            mae_bil_s.append(mae)
            print('mae: ', mae)
            psnr = compare_psnr(im_true, im_bil)
            psnr_bil_s.append(psnr)
            print('psnr:', psnr)

            ssim = compare_ssim(im_true, im_bil)
            ssim_bil_s.append(ssim)
            print('ssim:', ssim)
            del im_bil

        print('Loading fake_B:', fake_B_path)
        im_pred = np.load(fake_B_path).flatten()

        print('comparing normal...')
        rmse = compare_rmse(im_true, im_pred)
        rmse_s.append(rmse)
        print('rmse: ', rmse)
        mae = compare_mae(im_true, im_pred)
        mae_s.append(mae)
        print('mae: ', mae)
        psnr = compare_psnr(im_true, im_pred)
        psnr_s.append(psnr)
        print('psnr:', psnr)
        # ssim = compare_ssim(im_true, im_pred)
        ssim = 0
        ssim_s.append(ssim)
        print('ssim:', ssim)


        del im_pred
        del im_true

    print('\nReal Results\n')
    print('Total     RMSE: {:.4f}\t{:.4f}\n'.format(np.mean(rmse_s), np.std(rmse_s)))
    print('Total     MAE: {:.4f}\t{:.4f}\n'.format(np.mean(mae_s), np.std(mae_s)))
    print('Total     PSNR: {:.4f}\t{:.4f}\n'.format(np.mean(psnr_s), np.std(psnr_s)))
    print('Total     SSIM: {:.4f}\t{:.4f}\n'.format(np.mean(ssim_s), np.std(ssim_s)))
    f.write('\nReal Results\n')
    f.write('Total     RMSE: {:.4f}\t{:.4f}\n'.format(np.mean(rmse_s), np.std(rmse_s)))
    f.write('Total     MAE: {:.4f}\t{:.4f}\n'.format(np.mean(mae_s), np.std(mae_s)))
    f.write('Total     PSNR: {:.4f}\t{:.4f}\n'.format(np.mean(psnr_s), np.std(psnr_s)))
    f.write('Total     SSIM: {:.4f}\t{:.4f}\n'.format(np.mean(ssim_s), np.std(ssim_s)))

    # print('\nBIL Real Results\n')
    # print('Total     RMSE: {:.4f}\t{:.4f}\n'.format(np.mean(rmse_bil_s), np.std(rmse_bil_s)))
    # print('Total     MAE: {:.4f}\t{:.4f}\n'.format(np.mean(mae_bil_s), np.std(mae_bil_s)))
    # print('Total     PSNR: {:.4f}\t{:.4f}\n'.format(np.mean(psnr_bil_s), np.std(psnr_bil_s)))
    # print('Total     SSIM: {:.4f}\t{:.4f}\n'.format(np.mean(ssim_bil_s), np.std(ssim_bil_s)))
    # f.write('\nBIL Real Results\n')
    # f.write('Total     RMSE: {:.4f}\t{:.4f}\n'.format(np.mean(rmse_bil_s), np.std(rmse_bil_s)))
    # f.write('Total     MAE: {:.4f}\t{:.4f}\n'.format(np.mean(mae_bil_s), np.std(mae_bil_s)))
    # f.write('Total     PSNR: {:.4f}\t{:.4f}\n'.format(np.mean(psnr_bil_s), np.std(psnr_bil_s)))
    # f.write('Total     SSIM: {:.4f}\t{:.4f}\n'.format(np.mean(ssim_bil_s), np.std(ssim_bil_s)))

