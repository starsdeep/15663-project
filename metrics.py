'''
    Calculate PSNR and SSIM metrics
'''

import argparse
import glob
import os.path as path
import skimage.io
import skimage.measure
import numpy as np


def process(args):
    fns = glob.glob(path.join(args.imgdir, '*gt.jpg'))
    n = len(fns)
    psnr_list = []
    ssim_list = []
    for i in range(n):
        fn = fns[i]             # ground truth
        gt_img = skimage.io.imread(fn)
        fn = fn[:-6] + 'out.jpg'   # predicted
        pred_img = skimage.io.imread(fn)
        psnr = skimage.measure.compare_psnr(gt_img, pred_img)
        ssim = skimage.measure.compare_ssim(gt_img, pred_img, multichannel=True)
        psnr_list.append(psnr)
        ssim_list.append(ssim)
        log = '[%4d / %4d] %s PSNR: %8.3f SSIM: %8.3f' \
              % (i, n, path.basename(fn), psnr, ssim)
        print(log)

    print('mean PSNR: %.3f' % np.mean(psnr_list))
    print('mean SSIM: %.3f' % np.mean(ssim_list))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculating metrics...')
    parser.add_argument('--imgdir', type=str, required=True)
    args = parser.parse_args()
    process(args)