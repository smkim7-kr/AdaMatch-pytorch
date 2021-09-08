import time
import torch
import torch.nn.functional as F
import numpy as np

from tqdm import tqdm
from src.utils import AverageMeter, accuracy

def train(args, model, source_weak_dataloader, source_strong_dataloader, target_weak_dataloader, target_strong_dataloader,
          source_criterion, target_criterion, optimizer, scheduler, epoch, steps_per_epoch):
  dataset_loader = zip(source_weak_dataloader, source_strong_dataloader, target_weak_dataloader, target_strong_dataloader)
  epoch_start = time.time()
  trn_losses = AverageMeter()
  adjust_losses = AverageMeter()
  src_trn_acc = AverageMeter()
  tgt_trn_acc = AverageMeter()
  total_steps = args.epochs * steps_per_epoch
  model.train()
  tq = tqdm(enumerate(dataset_loader), total=steps_per_epoch)
  for n_step, ((data_source_weak, labels_source), (data_source_strong, _), (data_target_weak, labels_target), (data_target_strong, _)) in tq:
    data_source_weak, labels_source, data_source_strong = data_source_weak.to(args.device), labels_source.to(args.device), data_source_strong.to(args.device)
    data_target_weak, labels_target, data_target_strong = data_target_weak.to(args.device), labels_target.to(args.device), data_target_strong.to(args.device)

    assert data_source_weak.size(0) * args.uratio == data_target_weak.size(0)
    
    source_batch = torch.cat([data_source_weak, data_source_strong])
    all_batch = torch.cat([data_source_weak, data_source_strong, data_target_weak, data_target_strong])

    optimizer.zero_grad()
    all_logits = model(all_batch)
    all_logits_source = all_logits[:args.batch_size*2]

    model.eval()
    source_logits = model(source_batch)
    model.train()

    # Random logit interpolation
    lam = torch.rand(args.batch_size*2, args.num_classes).to(args.device)
    logits_source = lam * all_logits_source + (1-lam) * source_logits

    # Distribution alignment
    logits_source_weak = logits_source[:args.batch_size]
    pseudo_source = F.softmax(logits_source_weak, 0)

    logits_target = all_logits[args.batch_size*2:]
    logits_target_weak = logits_target[:args.batch_size*args.uratio]
    pseudo_target = F.softmax(logits_target_weak, 0)

    expect_ratio = torch.mean(pseudo_source) / torch.mean(pseudo_target)
    final_pseudolabels = F.normalize(pseudo_target*expect_ratio)

    # Relative confidence threshold
    pseudo_source_max = torch.max(pseudo_source, dim=1)[0]
    c_tau = args.tau * torch.mean(pseudo_source_max, 0)

    final_pseudolabels_max, final_pseudolabels_cls = torch.max(final_pseudolabels, dim=1)
    mask = final_pseudolabels_max >= c_tau

    source_loss = source_criterion(logits_source_weak ,labels_source) + source_criterion(logits_source[args.batch_size:] ,labels_source)
    pseudolabels = final_pseudolabels_cls.detach() #stop_gradient()
    target_loss = torch.mean(target_criterion(logits_target[args.uratio*args.batch_size:], pseudolabels) * mask, 0)
    
    PI = torch.tensor(np.pi).to(args.device)
    mu = 0.5 - torch.cos(torch.minimum(PI, (2*PI*(n_step+steps_per_epoch*epoch)) / total_steps)) / 2
    total_loss = source_loss + mu * target_loss

    acc_s = accuracy(logits_source_weak ,labels_source) 
    acc_t = accuracy(final_pseudolabels ,labels_target) 
    trn_losses.update(total_loss.item(), data_source_weak.size(0))
    adjust_losses.update(total_loss.item() / (1+mu), data_source_weak.size(0))
    src_trn_acc.update(acc_s.item(), data_source_weak.size(0))
    tgt_trn_acc.update(acc_t.item(), data_target_weak.size(0))

    total_loss.backward()
    optimizer.step()

    if (n_step+1) % 100 == 0:
      tq.set_description('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Adj Loss: {:.4f}'.format(
          epoch+1, args.epochs, n_step+1, steps_per_epoch, trn_losses.avg, adjust_losses.avg))

  print(f'----------------Epoch {epoch+1} train finished------------------')
  print('Epoch [{}/{}], Time elapsed: {:.4f}s, Loss: {:.4f}, Adj Loss: {:.4f}, Source acc: {:.4f}%, Target acc: {:.4f}%, learning rate: {:.6f}, mu: {:.6f}'.format(
        epoch+1, args.epochs, time.time()-epoch_start, trn_losses.avg, adjust_losses.avg, src_trn_acc.avg, tgt_trn_acc.avg, optimizer.param_groups[0]["lr"], mu))
  scheduler.step()
  return trn_losses.avg, adjust_losses.avg, src_trn_acc.avg, tgt_trn_acc.avg, mu
