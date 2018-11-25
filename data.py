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

        self.fns = glob.glob(gt_dir + '0*.ARW') # file names

        self.ids = [int(os.path.basename(fn)[0:5]) for fn in self.fns]

        # Raw data takes long time to load. Keep them in memory after loaded.
        self.gt_images = [None] * 6000
        self.input_images = dict()
        self.input_images['300'] = [None] * len(self.ids)
        self.input_images['250'] = [None] * len(self.ids)
        self.input_images['100'] = [None] * len(self.ids)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, ind):
        id = self.ids[ind]
        in_files = glob.glob(self.input_dir + '%05d_00*.ARW' % id)
        in_path = in_files[np.random.random_integers(0, len(in_files) - 1)]
        in_fn = os.path.basename(in_path)

        gt_files = glob.glob(self.gt_dir + '%05d_00*.ARW' % id)
        gt_path = gt_files[0]
        gt_fn = os.path.basename(gt_path)

        in_exposure = float(in_fn[9:-5])
        gt_exposure = float(gt_fn[9:-5])
        ratio = min(gt_exposure / in_exposure, 300)

        if self.input_images[str(ratio)[0:3]][ind] is None:
            raw = rawpy.imread(in_path)
            self.input_images[str(ratio)[0:3]][ind] = np.expand_dims(pack_raw(raw), axis=0) * ratio
            gt_raw = rawpy.imread(gt_path)
            im = gt_raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
            self.gt_images[ind] = np.expand_dims(np.float32(im / 65535.0), axis=0)

        # crop
        H = self.input_images[str(ratio)[0:3]][ind].shape[1]
        W = self.input_images[str(ratio)[0:3]][ind].shape[2]
        xx = np.random.randint(0, W - self.ps)
        yy = np.random.randint(0, H - self.ps)
        input_patch = self.input_images[str(ratio)[0:3]][ind][:, yy:yy + self.ps, xx:xx + self.ps, :]
        gt_patch = self.gt_images[ind][:, yy * 2:yy * 2 + self.ps * 2, xx * 2:xx * 2 + self.ps * 2, :]

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
            input_patch = np.transpose(input_patch, (0, 2, 1, 3)).copy()
            gt_patch = np.transpose(gt_patch, (0, 2, 1, 3)).copy()

        input_patch = np.minimum(input_patch, 1.0)

        input_patch = torch.from_numpy(input_patch)
        input_patch = torch.squeeze(input_patch)
        input_patch = input_patch.permute(2, 0, 1)
        
        gt_patch = torch.from_numpy(gt_patch)
        gt_patch = torch.squeeze(gt_patch)
        gt_patch = gt_patch.permute(2, 0, 1)
        
        return input_patch, gt_patch, id, ratio


class SonyTestDataset(Dataset):
    def __init__(self, input_dir, gt_dir):
        self.input_dir = input_dir
        self.gt_dir = gt_dir

        self.fns = glob.glob(input_dir + '1*.ARW')  # file names, 1 for testing.

        self.ids = [int(os.path.basename(fn)[0:5]) for fn in self.fns]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, ind):
        # input
        id = self.ids[ind]
        in_path = self.fns[ind]
        in_fn = os.path.basename(in_path)
        # ground truth
        gt_files = glob.glob(self.gt_dir + '%05d_00*.ARW' % id)
        gt_path = gt_files[0]
        gt_fn = os.path.basename(gt_path)
        # ratio
        in_exposure = float(in_fn[9:-5])
        gt_exposure = float(gt_fn[9:-5])
        ratio = min(gt_exposure / in_exposure, 300)

        # load images
        raw = rawpy.imread(in_path)
        input_full = np.expand_dims(pack_raw(raw), axis=0) * ratio

        im = raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
        # scale_full = np.expand_dims(np.float32(im/65535.0),axis = 0)*ratio
        scale_full = np.expand_dims(np.float32(im / 65535.0), axis=0)

        gt_raw = rawpy.imread(gt_path)
        im = gt_raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
        gt_full = np.expand_dims(np.float32(im / 65535.0), axis=0)

        # convert to tensor
        input_full = torch.from_numpy(input_full)
        input_full = torch.squeeze(input_full)
        input_full = input_full.permute(2, 0, 1)

        scale_full = torch.from_numpy(scale_full)
        scale_full = torch.squeeze(scale_full)

        gt_full = torch.from_numpy(gt_full)
        gt_full = torch.squeeze(gt_full)
        return input_full, scale_full, gt_full, id, ratio
