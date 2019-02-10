import numpy as np
from PIL import Image
import os

from scipy.ndimage.filters import convolve
import argparse 
from sklearn.metrics import mean_squared_error as compare_mse
from sklearn.metrics import mean_absolute_error as compare_mae
from skimage.measure import compare_psnr
from skimage.measure import compare_ssim

compare_mae5 = lambda a, b: compare_mae(a, b) * 1e5
compare_rmse5 = lambda a, b: np.sqrt(compare_mse(a, b)) * 1e5
norm_std = 1.0

parser = argparse.ArgumentParser()
parser.add_argument("--result_dir", type=str, required=True, help="Path to the generated images to be evaluated")
parser.add_argument("--version", type=str, default='testcube3d_latest', help="Version of model")
args = parser.parse_args()

path = os.path.join(args.result_dir, args.version)
output_file = os.path.join(args.result_dir, args.version, 'scores.txt')


def denorm(scan):
    out = np.zeros_like(scan)

    # Normalize (with real mean/std)
    channels =  [0, 1, 2, 3, 4, 5]
    channel_means = [0.000895749695547689, 6.744128981530014e-05, 6.852997960417064e-05, 0.0009213003187914121, 6.420715397339083e-05, 0.0008985036854860024]
    channel_stds =  [0.0004428829448270841, 7.248642579947666e-05, 7.538480811062319e-05, 0.0004398180675546679, 6.921599523395867e-05, 0.00044269766279459534]

    for channel, channel_mean, channel_std in zip(channels, channel_means, channel_stds):
        # out[channel] = (scan[channel] ** 2) * channel_mean
        out[channel] = scan[channel] * channel_std + channel_mean

    return out


with open(output_file, 'w') as f:
    # Load data
    real_As = sorted([os.path.join(path, x) for x in os.listdir(path) if 'real_A' in x])
    fake_Bs = sorted([os.path.join(path, x) for x in os.listdir(path) if 'fake_B' in x])
    real_Bs = sorted([os.path.join(path, x) for x in os.listdir(path) if 'real_B' in x])
    fake_As = sorted([os.path.join(path, x) for x in os.listdir(path) if 'fake_A' in x])

    is_backward = True if len(fake_As) > 0 else False
    if is_backward:
        real_As = sorted([os.path.join(path, x) for x in os.listdir(path) if 'real_A' in x])

    f.write('Model: {}\tVersion: {}\tFwrd: {}  \tBwrd: {}\n'.format(args.result_dir, args.version, len(fake_Bs), len(fake_As)))

    # Forward
    rmse_s_tot, mae_s_tot = [], []
    rmse_s_in, mae_s_in = [], []
    rmse_s_ex, mae_s_ex = [], []
    for fake_B_path, real_B_path in zip(fake_Bs, real_Bs):
        im_pred = np.load(fake_B_path) * norm_std
        im_true = np.load(real_B_path) * norm_std

        # Calculate 3d Interior and total masks
        r = 5       # <-- interior kernel radius
        thres0 = 0  # <-- allowed 0s in kernel

        # Denormalize
        im_pred = denorm(im_pred)
        im_true = denorm(im_true)

        print(im_true.mean(), im_true.std(), (im_true == 0).sum(), (im_true != 0).sum())
        sumbrain = np.repeat(np.absolute(im_true).sum(axis=0, keepdims=True), 6, axis=0)
        mask_total = sumbrain > 1e-7
        mask_interior = convolve(1 * mask_total, np.ones((1, r, r, r))) >= r ** 3
        mask_exterior = mask_total ^ mask_interior

        im_pred_tot = im_pred[mask_total]
        im_true_tot = im_true[mask_total]
        im_pred_in = im_pred[mask_interior]
        im_true_in = im_true[mask_interior]
        im_pred_ex = im_pred[mask_exterior]
        im_true_ex = im_true[mask_exterior]


        print('Interior  mask:', mask_interior.sum())
        print('Exterior  mask:', mask_exterior.sum())
        print('Total     mask:', mask_total.sum())


        # Calculate pixel-difference scores
        rmse_s_tot.append(compare_rmse5(im_true_tot.flatten(), im_pred_tot.flatten()))
        mae_s_tot.append(compare_mae5(im_true_tot.flatten(), im_pred_tot.flatten()))
        rmse_s_in.append(compare_rmse5(im_true_in.flatten(), im_pred_in.flatten()))
        mae_s_in.append(compare_mae5(im_true_in.flatten(), im_pred_in.flatten()))
        rmse_s_ex.append(compare_rmse5(im_true_ex.flatten(), im_pred_ex.flatten()))
        mae_s_ex.append(compare_mae5(im_true_ex.flatten(), im_pred_ex.flatten()))

    print('\nReal Results\n')
    print('Interior  RMSE: {:.4f}\t{:.4f}\n'.format(np.mean(rmse_s_in), np.std(rmse_s_in)))
    print('Exterior  RMSE: {:.4f}\t{:.4f}\n'.format(np.mean(rmse_s_ex), np.std(rmse_s_ex)))
    print('Total     RMSE: {:.4f}\t{:.4f}\n'.format(np.mean(rmse_s_tot), np.std(rmse_s_tot)))
    f.write('\nReal Results\n')
    f.write('Interior  RMSE: {:.4f}\t{:.4f}\n'.format(np.mean(rmse_s_in), np.std(rmse_s_in)))
    f.write('Exterior  RMSE: {:.4f}\t{:.4f}\n'.format(np.mean(rmse_s_ex), np.std(rmse_s_ex)))
    f.write('Total     RMSE: {:.4f}\t{:.4f}\n'.format(np.mean(rmse_s_tot), np.std(rmse_s_tot)))
