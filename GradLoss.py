# Computing L1 loss on gradient domain.
import torch
import torch.nn as nn
import torch.nn.functional as F


class GradLoss(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.kernel = torch.FloatTensor([[0., -1., 0.],
                                         [-1., 0., 1.],
                                         [0., 1., 0.]])
        self.kernel = self.kernel.view(1, 1, 3, 3).to(device)
        self.criterion = nn.L1Loss()

    def forward(self, pred, target):
        pred = pred.view(-1, 1, pred.size(2), pred.size(3))
        grad_pred = F.conv2d(pred, self.kernel, padding=1)
        grad_pred = grad_pred.view(-1, 3, grad_pred.size(2), grad_pred.size(3))

        target = target.view(-1, 1, target.size(2), target.size(3))
        grad_target = F.conv2d(target, self.kernel, padding=1)
        grad_target = grad_target.view(-1, 3, grad_target.size(2), grad_target.size(3))

        return self.criterion(grad_pred, grad_target)
