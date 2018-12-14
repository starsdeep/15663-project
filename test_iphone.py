import numpy as np
import glob
import rawpy
import argparse
import os.path as path
import torch
from models import Unet
import scipy.io


def crop_center(img, cropx, cropy):
    y, x = img.shape[1], img.shape[2]
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)
    return img[:, starty:starty+cropy, startx:startx+cropx, :]


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


def test(args):
    # device
    device = torch.device("cuda:%d" % args.gpu if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    # images path
    fns = glob.glob(path.join(args.imgdir, '*.DNG'))
    n = len(fns)

    # model
    model = Unet()
    model.load_state_dict(torch.load(args.model))
    model.to(device)
    model.eval()

    # ratio
    ratio = 200

    for idx in range(n):
        fn = fns[idx]
        print(fn)
        raw = rawpy.imread(fn)

        input = np.expand_dims(pack_raw(raw), axis=0) * ratio
        scale_full = np.expand_dims(np.float32(input / 65535.0), axis=0)
        input = crop_center(input, 1024, 1024)
        input = torch.from_numpy(input)
        input= torch.squeeze(input)
        input = input.permute(2, 0, 1)
        input = torch.unsqueeze(input, dim=0)
        input = input.to(device)

        outputs = model(input)
        outputs = outputs.cpu().detach()
        outputs = torch.squeeze(outputs)
        outputs = outputs.permute(1, 2, 0)

        outputs = outputs.numpy()
        outputs = np.minimum(np.maximum(outputs, 0), 1)

        scale_full = torch.from_numpy(scale_full)
        scale_full = torch.squeeze(scale_full)

        scipy.misc.toimage(outputs * 255, high=255, low=0, cmin=0, cmax=255).save(
            path.join(args.imgdir, path.basename(fn) + '_out.jpg'))
        # scipy.misc.toimage(scale_full * 255, high=255, low=0, cmin=0, cmax=255).save(
        #     path.join(args.imgdir, path.basename(fn) + '_scale.jpg'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='test .dng images')
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--imgdir', type=str, required=True)
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()

    test(args)