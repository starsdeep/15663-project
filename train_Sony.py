# uniform content loss + adaptive threshold + per_class_input + recursive G
# improvement upon cqf37
from __future__ import division
import os, scipy.io
import numpy as np
import logging
import argparse
import sys
from data import SonyDataset
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from models import Unet
from datetime import datetime
from GradLoss import GradLoss

def train(args):
    # device
    device = torch.device("cuda:%d" % args.gpu if torch.cuda.is_available() else "cpu")

    # data
    trainset = SonyDataset(args.input_dir, args.gt_dir, args.ps)
    train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=12)
    logging.info("data loading okay")

    # model
    model = Unet().to(device)

    # resume
    starting_epoch = 0
    if args.resume is not None:
        model.load_state_dict(torch.load(args.resume))
        starting_epoch = int(args.resume[-7:-3])
        print('resume at %d epoch' % starting_epoch)


    # loss function
    color_loss = nn.L1Loss()
    gradient_loss = GradLoss(device)

    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

    # lr scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.1)

    # training
    running_loss = 0.0
    for epoch in range(starting_epoch+1, starting_epoch + args.num_epoch):
        scheduler.step()
        for i, databatch in enumerate(train_loader):
            # get the inputs
            input_patch, gt_patch, train_id, ratio = databatch
            input_patch, gt_patch = input_patch.to(device), gt_patch.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(input_patch)
            loss = color_loss(outputs, gt_patch) + gradient_loss(outputs, gt_patch)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % args.log_interval == (args.log_interval - 1):
                print('[%d, %5d] loss: %.3f %s' %
                      (epoch, i, running_loss / args.log_interval, datetime.now()))
                running_loss = 0.0

            if epoch % args.save_freq == 0:
                if not os.path.isdir(os.path.join(args.result_dir, '%04d' % epoch)):
                    os.makedirs(os.path.join(args.result_dir, '%04d' % epoch))
                
                gt_patch = gt_patch.cpu().detach().numpy()
                outputs = outputs.cpu().detach().numpy()
                train_id = train_id.numpy()
                ratio = ratio.numpy()

                temp = np.concatenate((gt_patch[0, :, :, :], outputs[0, :, :, :]), axis=2)
                scipy.misc.toimage(temp * 255, high=255, low=0, cmin=0, cmax=255).save(
                    args.result_dir + '%04d/%05d_00_train_%d.jpg' % (epoch, train_id[0], ratio[0]))

        # at the end of epoch
        if epoch % args.model_save_freq == 0:
            torch.save(model.state_dict(), args.checkpoint_dir + './model_%d.pl' % epoch)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="command for training p3d network")
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--input_dir', type=str, default='./dataset/Sony/short/')
    parser.add_argument('--gt_dir', type=str, default='./dataset/Sony/long/')
    parser.add_argument('--checkpoint_dir', type=str, default='./result_grad_loss/')
    parser.add_argument('--result_dir', type=str, default='./result_grad_loss/')
    parser.add_argument('--ps', type=int, default=512)
    parser.add_argument('--log_interval', type=int, default=10)
    parser.add_argument('--save_freq', type=int, default=500)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--wd', type=float, default=0)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_epoch', type=int, default=2000)
    parser.add_argument('--model_save_freq', type=int, default=1)
    parser.add_argument('--resume', type=str, help='continue training')
    args = parser.parse_args()

    # Create Output Dir
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)

    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)

    # Set Logger
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(message)s',
                        filename=os.path.join(args.result_dir, 'log.txt'),
                        filemode='w')
    # Define a new Handler to log to console as well
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    logging.getLogger('').addHandler(console)

    # Start training
    logging.info(" ".join(sys.argv))
    logging.info(args)
    logging.info("using device %s" % str(args.gpu))
    train(args)
