import argparse, os, sys

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data as data

from tensorboardX import SummaryWriter

from data_loader import load_dataloader
from models import load_model
from src.utils import *
from src.optimizer import make_optimizer
from src.scheduler import make_scheduler
from train import train
from tqdm import tqdm

parser = argparse.ArgumentParser(description='AdaMatch Pytorch')

parser.add_argument('--epochs', default=1024, type=int)
parser.add_argument('--batch_size', default=12, type=int)
parser.add_argument('--num-workers', default=0, type=int)
parser.add_argument('--lr', '--learning-rate', default=0.002, type=float)
parser.add_argument('--seed', default=2021, type=int)
parser.add_argument('--device', default='cuda')
parser.add_argument('--checkpoint-dir', type=str, default='checkpoint')
parser.add_argument('--checkpoint-name', type=str, default='')

# parser.add_argument('--datasets', default='CIFAR10', choices=('CIFAR10', 'CIFAR100', 'SVHN', 'STL10'))
parser.add_argument('--num-classes', default=10, type=int)
parser.add_argument('--num-labels', default=400, type=int)
parser.add_argument('--evaluate', action='store_true', default=False)
parser.add_argument('--weight-decay', type=float, default=5e-4)
parser.add_argument('--optimizer', default='ADAM', choices=('SGD', 'ADAM', 'ADAMW', 'RADAM', 'LOOKAHEAD'))
parser.add_argument('--decay-type', default='cosine_warmup', choices=('step', 'step_warmup', 'cosine_warmup'))

#Hyperparameters
parser.add_argument('--N', default=3, type=int)
parser.add_argument('--M', default=3, type=int)
parser.add_argument('--uratio', default=3, type=int)
parser.add_argument('--tau', default=0.9, type=float)

def main():
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    source_weak_dataloader, source_strong_dataloader, source_test_dataloader, target_weak_dataloader, target_strong_dataloader, target_test_dataloader = load_dataloader(args)
    
    model = load_model(args)
    model = model.to(args.device)

    source_criterion = nn.CrossEntropyLoss().to(args.device)
    target_criterion = nn.CrossEntropyLoss(reduction='none').to(args.device)
    optimizer = make_optimizer(args, model)
    scheduler = make_scheduler(args, optimizer)

    if not os.path.isdir(args.checkpoint_dir):                                                           
        os.mkdir(args.checkpoint_dir)

    writer = SummaryWriter('result')
    steps_per_epoch = min(len(source_weak_dataloader), len(target_weak_dataloader))
    best_loss = 10e10
    tq = tqdm(range(args.epochs), total=args.epochs)
    for epoch in tq:
        trn_loss, adjust_losses, src_trn_acc, tgt_trn_acc, mu = train(args, model, source_weak_dataloader, source_strong_dataloader, 
        target_weak_dataloader, target_strong_dataloader, source_criterion, target_criterion, optimizer, scheduler, epoch, steps_per_epoch)
        writer.add_scalar('losses/train_loss', trn_loss, epoch+1)
        writer.add_scalar('losses/adjust_train_loss', adjust_losses, epoch+1)
        writer.add_scalar('accs/source_train_accuracy', src_trn_acc, epoch+1)
        writer.add_scalar('accs/target_train_accuracy', tgt_trn_acc, epoch+1)
        writer.add_scalar('params/learning_rate', optimizer.param_groups[0]["lr"], epoch+1)
        writer.add_scalar('params/mu', mu, epoch+1)

        if trn_loss < best_loss:
            save_checkpoint(model, args.checkpoint_dir, epoch+1, file_name='adamatch.pt')
            best_loss = trn_loss

    
if __name__ == '__main__':
    main()