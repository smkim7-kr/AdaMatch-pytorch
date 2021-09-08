import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import random, os

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

class AverageMeter(object):
    '''
    Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    '''
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    batch_size = target.size(0)
    _, pred = output.topk(1, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    correct_k = correct.reshape(-1).float().sum(0)
    return correct_k.mul_(100.0 / batch_size)

class SmoothCrossEntropyLoss(nn.Module):
    def __init__(self, label_smoothing=0.0):
        super().__init__()
        self.label_smoothing = label_smoothing

    def forward(self, input, target):
        if len(target.size()) == 1:
            target = torch.nn.functional.one_hot(target, num_classes=input.size(-1))
            target = target.float().cuda()
        if self.label_smoothing > 0.0:
            s_by_c = self.label_smoothing / len(input[0])
            smooth = torch.zeros_like(target)
            smooth = smooth + s_by_c
            target = target * (1. - s_by_c) + smooth

        return torch.nn.cross_entropy(input, target)

def save_checkpoint(model, saved_dir, epoch, file_name):
    file_name = "Epoch" + str(epoch) + "_" + file_name
    check_point = {'net': model.state_dict()}
    output_path = os.path.join(saved_dir, file_name)
    torch.save(model, output_path)