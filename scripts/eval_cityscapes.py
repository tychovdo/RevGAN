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

# Load model 
#caffe.set_mode_cpu();
caffe.set_device(0)
caffe.set_mode_gpu()

net = caffe.Net(args.caffemodel_dir + '/deploy.prototxt',
                args.caffemodel_dir + 'fcn-8s-cityscapes.caffemodel',
                caffe.TEST)
                

def color2id(color_image):
    _, colors = zip(*[(x.id, x.color) for x in labels if not x.ignoreInEval])
    colors = np.array(colors)
    tree = KDTree(colors)
    _, id_image = tree.query(color_image)
    return id_image.astype(np.uint8)
    
def id2color(id_image):
    _, colors = zip(*[(x.id, x.color) for x in labels if not x.ignoreInEval])
    color_image = np.array(colors, dtype=np.uint8)[id_image]
    return color_image

def get_scores(hist):
    # Source: https://github.com/phillipi/pix2pix/tree/master/scripts/eval_cityscapes
    
    # Mean pixel accuracy
    acc = np.diag(hist).sum() / (hist.sum() + 1e-12)

    # Per class accuracy
    cl_acc = np.diag(hist) / (hist.sum(1) + 1e-12)

    # Per class IoU
    iu = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist) + 1e-12)

    return acc, np.nanmean(cl_acc), np.nanmean(iu), cl_acc, iu
    
def preprocess_image(im):
    in_ = np.array(im, dtype=np.float32)
    in_ = in_[:, :, ::-1]
    in_ -= np.array((72.78044, 83.21195, 73.45286), dtype=np.float32)
    in_ = in_.transpose((2, 0, 1))
    return in_

def segment_image(im):
    print('im', im.shape)
    resized = scipy.misc.imresize(im, (1024, 2048, 3))
    print('re', resized.shape)
    return segrun(net, preprocess_image(resized))
    

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
        n_cl = 19
        hist_perframe = np.zeros((n_cl, n_cl))
        for i, (fakeb_path, realb_path) in enumerate(zip(fake_Bs, real_Bs)):
            print(i, len(fake_Bs))
            fakeb = np.array(Image.open(fakeb_path))
            realb = np.array(Image.open(realb_path))
            hist_perframe += confusion_matrix(color2id(fakeb).flatten(), color2id(realb).flatten(), labels=range(n_cl))

        per_pixel_acc, per_class_acc, class_iou, _, _ = get_scores(hist_perframe)

        f.write('\nCLASSIFICATION SCORES\n')
        f.write('Per Pixel acc.: %s\n' % per_pixel_acc)
        f.write('Per-class acc.: %s\n' % per_class_acc)
        f.write('Class IOU:      %s\n' % class_iou)
    
    # Backward
    if is_backward:
        print('backward...')
        n_cl = 19
        hist_perframe = np.zeros((n_cl, n_cl))
        for i, (fakea_path, realb_path) in enumerate(zip(fake_As, real_Bs)):
            print(i, len(fake_As))
            
            fakea = np.array(Image.open(fakea_path))
            realb = np.array(Image.open(realb_path))
            
            print(fakea.shape)
            print('fwd')
            fakea_segmented = segment_image(fakea)

            print('res')
            y_pred = scipy.misc.imresize(fakea_segmented, (128, 128))
            y_true = color2id(realb)

            print('conf')
            hist_perframe += confusion_matrix(y_pred.flatten(), y_true.flatten(), labels=range(n_cl))

        per_pixel_acc, per_class_acc, class_iou, _, _ = get_scores(hist_perframe)

        f.write('\nFCN-SCORES\n')
        f.write('Per Pixel acc.: %s\n' % per_pixel_acc)
        f.write('Per-class acc.: %s\n' % per_class_acc)
        f.write('Class IOU:      %s\n' % class_iou)
