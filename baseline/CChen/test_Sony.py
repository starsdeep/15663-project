import argparse
import os
import torch
from data import SonyTestDataset
from torch.utils.data import DataLoader
from models import Unet
import scipy.io
from tqdm import tqdm
import numpy as np


def test(args):
    # device
    device = torch.device("cuda:%d" % args.gpu if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    # data
    testset = SonyTestDataset(args.input_dir, args.gt_dir)
    test_loader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # model
    model = Unet()
    model.load_state_dict(torch.load(args.model))
    model.to(device)
    model.eval()

    # testing
    for i, databatch in tqdm(enumerate(test_loader), total=len(test_loader)):
        input_full, scale_full, gt_full, test_id, ratio = databatch
        scale_full, gt_full = torch.squeeze(scale_full), torch.squeeze(gt_full)

        # processing
        inputs = input_full.to(device)
        outputs = model(inputs)
        outputs = outputs.cpu().detach()
        outputs = torch.squeeze(outputs)
        outputs = outputs.permute(1, 2, 0)

        # scaling can clipping
        outputs, scale_full, gt_full = outputs.numpy(), scale_full.numpy(), gt_full.numpy()
        scale_full = scale_full * np.mean(gt_full) / np.mean(
            scale_full)  # scale the low-light image to the same mean of the ground truth
        outputs = np.minimum(np.maximum(outputs, 0), 1)

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
    parser = argparse.ArgumentParser(description="evaluating model")
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--input_dir', type=str, default='../../dataset/Sony/short/')
    parser.add_argument('--gt_dir', type=str, default='../../dataset/Sony/long/')
    parser.add_argument('--result_dir', type=str, default='../../result_Sony/')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=8, help='multi-threads for data loading')
    parser.add_argument('--model', type=str, required=True)
    args = parser.parse_args()

    # Create Output Dir
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)

    test(args)
