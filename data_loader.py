import torch
import torchvision
import torch.utils.data as data
import torchvision.transforms as transforms
from RandAugment import RandAugment

import numpy as np

#transformations
def data_transforms(args, type):
    if type=='weak':
        transforms_ = transforms.Compose([
            transforms.Lambda(lambda image: image.convert('RGB')),                              
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.Resize(32),
            transforms.ToTensor()
        ])
    elif type=='strong':
        transforms_ = transforms.Compose([
            transforms.Lambda(lambda image: image.convert('RGB')),                                  
            RandAugment(args.N, args.M), 
            transforms.RandomCrop(32, padding=4),
            transforms.Resize(32),
            transforms.ToTensor()
        ])

    elif type=='test':
        transforms_ = transforms.Compose([
            transforms.Lambda(lambda image: image.convert('RGB')),
            transforms.Resize(32),
            transforms.ToTensor()
        ])

    return transforms_

def load_dataloader(args):
    source_weak_dataset = torchvision.datasets.SVHN(root='./data', split='train', download=True, transform=data_transforms(args, 'weak'))
    source_strong_dataset = torchvision.datasets.SVHN(root='./data', split='train', download=True, transform=data_transforms(args, 'strong'))
    source_test_dataset = torchvision.datasets.SVHN(root='./data', split='test', download=True, transform=data_transforms(args, 'test'))

    target_weak_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=data_transforms(args, 'weak'))
    target_strong_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=data_transforms(args, 'strong'))
    target_test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=data_transforms(args, 'test'))

    #Dataloader
    source_weak_dataloader = torch.utils.data.DataLoader(dataset=source_weak_dataset, 
                                                      batch_size=args.batch_size,
                                                      shuffle=False,
                                                      pin_memory=True,
                                                      num_workers=0,
                                                      drop_last=True)

    source_strong_dataloader = torch.utils.data.DataLoader(dataset=source_strong_dataset, 
                                                            batch_size=args.batch_size,
                                                            shuffle=False,
                                                            pin_memory=True,
                                                            num_workers=0,
                                                            drop_last=True)

    source_test_dataloader = torch.utils.data.DataLoader(dataset=source_test_dataset, 
                                                        batch_size=args.batch_size,
                                                        shuffle=True,
                                                        pin_memory=True,
                                                        num_workers=0)

    target_weak_dataloader = torch.utils.data.DataLoader(dataset=target_weak_dataset, 
                                                        batch_size=args.uratio*args.batch_size,
                                                        shuffle=False,
                                                        pin_memory=True,
                                                        num_workers=0,
                                                        drop_last=True)

    target_strong_dataloader = torch.utils.data.DataLoader(dataset=target_strong_dataset, 
                                                            batch_size=args.uratio*args.batch_size,
                                                            shuffle=False,
                                                            pin_memory=True,
                                                            num_workers=0,
                                                            drop_last=True)

    target_test_dataloader = torch.utils.data.DataLoader(dataset=target_test_dataset, 
                                                        batch_size=args.uratio*args.batch_size,
                                                        shuffle=True,
                                                        pin_memory=True,
                                                        num_workers=0)
                                            
    return source_weak_dataloader, source_strong_dataloader, source_test_dataloader, target_weak_dataloader, \
        target_strong_dataloader, target_test_dataloader