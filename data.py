import torch
from torch.utils.data import Dataset
import os
import numpy as np
import glob
import rawpy

def pack_raw(raw):
    # pack Bayer image to 4 channels
    im = raw.raw_image_visible.astype(np.float32)
    im = np.maximum(im - 512, 0) / (16383 - 512)  # subtract the black level

    im = np.expand_dims(im, axis=2)
    img_shape = im.shape
    H = img_shape[0]
    W = img_shape[1]

    out = np.concatenate((im[0:H:2, 0:W:2, :],
                          im[0:H:2, 1:W:2, :],
                          im[1:H:2, 1:W:2, :],
                          im[1:H:2, 0:W:2, :]), axis=2)
    return out

class SonyDataset(Dataset):

    def __init__(self, input_dir, gt_dir, ps=512):
        self.input_dir = input_dir
        self.gt_dir = gt_dir
        self.ps = ps

        self.train_fns = glob.glob(gt_dir + '0*.ARW')
        self.train_ids = [int(os.path.basename(train_fn)[0:5]) for train_fn in self.train_fns]

        # Raw data takes long time to load. Keep them in memory after loaded.
        self.gt_images = [None] * 6000
        self.input_images = {}
        self.input_images['300'] = [None] * len(self.train_ids)
        self.input_images['250'] = [None] * len(self.train_ids)
        self.input_images['100'] = [None] * len(self.train_ids)

    def __len__(self):
        return len(self.train_ids)

    def __getitem__(self, ind):
        train_id = self.train_ids[ind]
        in_files = glob.glob(self.input_dir + '%05d_00*.ARW' % train_id)
        in_path = in_files[np.random.random_integers(0, len(in_files) - 1)]
        in_fn = os.path.basename(in_path)

        gt_files = glob.glob(self.gt_dir + '%05d_00*.ARW' % train_id)
        gt_path = gt_files[0]
        gt_fn = os.path.basename(gt_path)

        in_exposure = float(in_fn[9:-5])
        gt_exposure = float(gt_fn[9:-5])
        ratio = min(gt_exposure / in_exposure, 300)

        if self.input_images[str(ratio)[0:3]][ind] is None:
            raw = rawpy.imread(in_path)
            raw = pack_raw(raw) * ratio
            # convert (H,W,C) to (C,H,W)
            self.input_images[str(ratio)[0:3]][ind] = np.transpose(raw, (2, 1, 0))


        if self.gt_images[ind] is None:
            gt_raw = rawpy.imread(gt_path)
            im = gt_raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
            im = np.float32(im / 65535.0)
            # convert (H,W,C) to (C,H,W)
            self.gt_images[ind] = np.transpose(im, (2, 1, 0))

        # crop
        H = self.input_images[str(ratio)[0:3]][ind].shape[0]
        W = self.input_images[str(ratio)[0:3]][ind].shape[1]
        xx = np.random.randint(0, W - self.ps)
        yy = np.random.randint(0, H - self.ps)
        input_patch = self.input_images[str(ratio)[0:3]][ind][:, yy:yy + self.ps, xx:xx + self.ps]
        gt_patch = self.gt_images[ind][:, yy * 2:yy * 2 + self.ps * 2, xx * 2:xx * 2 + self.ps * 2]

        # data augmentation
        # random flip vertically
        if np.random.randint(2, size=1)[0] == 1:
            input_patch = np.flip(input_patch, axis=1).copy()
            gt_patch = np.flip(gt_patch, axis=1).copy()
        # random flip horizontally
        if np.random.randint(2, size=1)[0] == 1:
            input_patch = np.flip(input_patch, axis=2).copy()
            gt_patch = np.flip(gt_patch, axis=2).copy()
        # random transpose
        if np.random.randint(2, size=1)[0] == 1:
            input_patch = np.transpose(input_patch, (0, 2, 1)).copy()
            gt_patch = np.transpose(gt_patch, (0, 2, 1)).copy()

        input_patch = np.minimum(input_patch, 1.0)

        return torch.from_numpy(input_patch), torch.from_numpy(gt_patch), train_id, ratio























