import argparse
import os
import torch
from data import SonyTestDataset
from torch.utils.data import DataLoader
import scipy.io
from tqdm import tqdm
import numpy as np
import pybm3d


def test(args):
    # data
    testset = SonyTestDataset(args.input_dir, args.gt_dir)
    test_loader = DataLoader(testset, batch_size=1, shuffle=False)

    # testing
    for i, databatch in tqdm(enumerate(test_loader), total=len(test_loader)):
        _, scale_full, gt_full, test_id, ratio = databatch
        scale_full, gt_full = torch.squeeze(scale_full), torch.squeeze(gt_full)

        # processing
        scale_full, gt_full = scale_full.numpy(), gt_full.numpy()
        input_full = scale_full

        output = pybm3d.bm3d.bm3d(input_full, 40)

        # scaling can clipping
        scale_full = scale_full * np.mean(gt_full) / np.mean(
            scale_full)  # scale the low-light image to the same mean of the ground truth
        outputs = np.minimum(np.maximum(output, 0), 1)

        # saving
        if not os.path.isdir(os.path.join(args.result_dir, 'eval')):
            os.makedirs(os.path.join(args.result_dir, 'eval'))
        scipy.misc.toimage(scale_full * 255, high=255, low=0, cmin=0, cmax=255).save(
            os.path.join(args.result_dir, 'eval', '%05d_00_train_%d_scale.jpg' % (test_id[0], ratio[0])))
        scipy.misc.toimage(outputs * 255, high=255, low=0, cmin=0, cmax=255).save(
            os.path.join(args.result_dir, 'eval', '%05d_00_train_%d_out.jpg' % (test_id[0], ratio[0])))
        scipy.misc.toimage(gt_full * 255, high=255, low=0, cmin=0, cmax=255).save(
            os.path.join(args.result_dir, 'eval', '%05d_00_train_%d_gt.jpg' % (test_id[0], ratio[0])))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="BM3D (OpenCV Version)")
    parser.add_argument('--input_dir', type=str, default='../../dataset/Sony/short/')
    parser.add_argument('--gt_dir', type=str, default='../../dataset/Sony/long/')
    parser.add_argument('--result_dir', type=str, default='./result_bm3d/')
    args = parser.parse_args()

    # Create Output Dir
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)

    test(args)
