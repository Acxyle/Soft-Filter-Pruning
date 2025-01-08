#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 18:34:31 2024

@author: acxyle


This code contains 2 Trainers that encapsule: (1) dataset; (2) dataloader; (3) Optimizer; (4) Scheduler; (5) Train & Val
    - Trainer_lite() is the simplified Trainer() inherited from spikingjelly
    - Trainer_basic() is the general training script from pytorch

"""


import os
import random
import numpy as np
from tqdm import tqdm
import time
import datetime
import argparse

# ---
import torch
import torch.nn as nn
import torchvision
from torch.utils.data.dataloader import default_collate
from torchvision.transforms.functional import InterpolationMode

# --- sp
from spikingjelly.activation_based import  functional
from spikingjelly.activation_based.model.tv_ref_classify import presets, transforms, utils


import shutil

import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchvision.datasets as datasets


# ----------------------------------------------------------------------------------------------------------------------
def universal_training_parser(parser):
    """
    the default values are specified for pytorch NN training
    """
    
    parser.add_argument("--device", default="cuda", type=str, help="device (Use cuda or cpu Default: cuda)")
    
    parser.add_argument("-j", "--workers", default=16, type=int, metavar="N", help="number of data loading workers (default: 16)")
    
    # --- basic
    parser.add_argument("--momentum", default=0.9, type=float, metavar="M", help="momentum")
    parser.add_argument("--wd", "--weight_decay", default=1e-4, type=float, metavar="W", dest="weight_decay")
    parser.add_argument("--norm_weight_decay", default=None, type=float)
    
    parser.add_argument('--lr_adjust', type=int, default=50, help='number of epochs that change learning rate')
    
    # --- lite
    parser.add_argument("--label_smoothing", default=0.1, type=float, help="label smoothing (default: 0.1)", dest="label_smoothing")
    
    parser.add_argument("--mixup_alpha", default=0.2, type=float, help="mixup alpha")
    parser.add_argument("--cutmix_alpha", default=0.2, type=float, help="cutmix alpha")
    
    parser.add_argument("--lr_scheduler", default="cosa", type=str, help="the lr scheduler")
    parser.add_argument("--lr_warmup_epochs", default=5, type=int, help="the number of epochs to warmup")
    parser.add_argument("--lr_warmup_method", default="linear", type=str, help="the warmup method")
    parser.add_argument("--lr_warmup_decay", default=0.01, type=float, help="the decay for lr")

    parser.add_argument("--start_epoch", default=0, type=int, metavar="N", help="start epoch")
    
    parser.add_argument("--auto_augment", default='ta_wide', type=str, help="auto augment policy (default: ta_wide)")
    parser.add_argument("--random_erase", default=0.1, type=float, help="random erasing probability (default: 0.1)")

    parser.add_argument("--interpolation", default="bilinear", type=str, help="the interpolation method (default: bilinear)")

    parser.add_argument("--val_resize_size", default=232, type=int, help="the resize size used for validation (default: 232)")
    parser.add_argument("--val_crop_size", default=224, type=int, help="the central crop size used for validation (default: 224)")
    parser.add_argument("--train_crop_size", default=176, type=int, help="the random crop size used for training (default: 176)")

    parser.add_argument("--seed", default=2020, type=int, help="the random seed")

    parser.add_argument("--disable_pinmemory", action="store_true", help="not use pin memory in dataloader, which can help reduce memory consumption")
    parser.add_argument("--disable_amp", default=True, help="not use automatic mixed precision training (set True when pruning)")
    
    return parser


def training_parser(add_help=True):

    parser = argparse.ArgumentParser(description="PyTorch Classification Training", add_help=add_help)
    parser = universal_training_parser(parser)
    
    return parser


# ----------------------------------------------------------------------------------------------------------------------
class Trainer_Base():
    
    def __init__(self, args):
        
        print(args)
        
        # --- prepare for the training config
        self.device = torch.device(args.device)
        self.criterion = nn.CrossEntropyLoss()
 
        # ----- prepare for the dataset
        self.dataset, self.dataset_val = self.prepare_datasets(args)
        self.prepare_dataloaders(args)
        
        # --- prepare for the model
        self.load_model(args)     # --- this must be defined outside
        self.model.to(self.device)
        
    
    def train(self,) -> None:
        
        raise AssertionError
    
    
    def train_one_epoch(self, ) -> None:
        
        raise AssertionError
    

    def validate(self, ) -> None:
        
        raise AssertionError
    
    
    def save_checkpoint(self, args, val_acc, epoch, optimizer):
        
        assert hasattr(self, 'best_pred_acc')
        
        self.filename = os.path.join(args.output_dir, f'checkpoint-{args.arch}-{args.timestamp}.pth.tar')
        self.bestname = os.path.join(args.output_dir, f'best-{args.arch}-{args.timestamp}.pth.tar')
        
        check_point = {
                        'epoch': epoch + 1,
                        'arch': args.arch,
                        'state_dict': self.model.state_dict(),
                        'best_prec1': self.best_pred_acc,
                        'optimizer': optimizer.state_dict(),
                        }
        
        torch.save(check_point, self.filename)
        
        if val_acc > self.best_pred_acc:
            
            shutil.copyfile(self.filename, self.bestname)
        
        self.best_pred_acc = max(val_acc, self.best_pred_acc)
        
    
    @staticmethod
    def print_log(print_string, log):
        print("{:}".format(print_string))
        log.write('[{:}] {:}\n'.format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), print_string))
        log.flush()

    
    @staticmethod
    def accuracy(output, target, topk=(1,)):
        """Computes the precision@k for the specified values of k"""
        
        if target.ndim == 2:
            target = target.max(dim=1)[1]
        
        maxk = max(topk)
        batch_size = target.size(0)
    
        _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))
    
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
    

class Trainer_basic(Trainer_Base):
    """
        ...
    """
    
    def __init__(self, args):
        
        super().__init__(args)
        
    
    @staticmethod
    def prepare_datasets(args):
        
        assert 'train' in os.listdir(args.data_path)     # assume the dataset is organized by ImageNet style
        
        traindir = os.path.join(args.data_path, 'train')
        valdir = os.path.join(args.data_path, 'val')
        
        normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        dataset_train = datasets.ImageFolder(
                                            traindir, torchvision.transforms.Compose([
                                                            torchvision.transforms.RandomResizedCrop(224),
                                                            torchvision.transforms.RandomHorizontalFlip(),
                                                            torchvision.transforms.ToTensor(),
                                                            normalize,
                                            ]))
        
        dataset_val = datasets.ImageFolder(
                                            valdir, torchvision.transforms.Compose([
                                                        torchvision.transforms.Resize(256),
                                                        torchvision.transforms.CenterCrop(224),
                                                        torchvision.transforms.ToTensor(),
                                                        normalize,
                                            ]))
        
        return dataset_train, dataset_val
        
    
    def prepare_dataloaders(self, args):
        
        self.train_loader = torch.utils.data.DataLoader(
                                            self.dataset, 
                                            batch_size=args.batch_size, 
                                            shuffle=True,
                                            num_workers=args.workers, 
                                            pin_memory=True, 
                                            sampler=None
                                            )

        self.val_loader = torch.utils.data.DataLoader(
                                            self.dataset_val,
                                            batch_size=args.batch_size, 
                                            shuffle=False,
                                            num_workers=args.workers, 
                                            pin_memory=True
                                            )
        
    @staticmethod
    def set_optimizer(args,  model):
        
        return torch.optim.SGD(model.parameters(), args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay,
                                    nesterov=True
                                )
    
    
    @staticmethod
    def set_lr_scheduler(args, optimizer):
        
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_adjust, gamma=0.1)
    

    def train_one_epoch(self, args, optimizer, epoch, log):
        
        self.model.train()

        top1 = AverageMeter()
        top5 = AverageMeter()
        _loss = AverageMeter()

        for i, (image, target) in enumerate(self.train_loader):
           
            image = image.to(self.device, non_blocking=True)
            target = target.to(self.device, non_blocking=True)
            
            output = self.model(image)
            loss = self.criterion(output, target)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # ---
            acc1, acc5 = self.accuracy(output.data, target, topk=(1, 5))
            batch_size = target.shape[0]
            
            top1.update(acc1.item(), batch_size)
            top5.update(acc5.item(), batch_size)
            _loss.update(loss.data.item(), batch_size)

        self.print_log(f'===== [{epoch}/{args.epochs}] Train ====> Prec@1 {top1.avg:.3f}% Prec@5 {top5.avg:.3f}%', log)


    def validate(self, args, log):
        
        self.model.eval()
        
        top1 = AverageMeter()
        top5 = AverageMeter()
        _loss = AverageMeter()

        with torch.inference_mode():

            for i, (image, target) in enumerate(self.val_loader):
                
                image = image.to(self.device, non_blocking=True)
                target = target.to(self.device, non_blocking=True)
                
                output = self.model(image)
                loss = self.criterion(output, target)
    
                acc1, acc5 = self.accuracy(output.data, target, topk=(1, 5))
                batch_size = target.shape[0]
                
                top1.update(acc1.item(), batch_size)
                top5.update(acc5.item(), batch_size)
                _loss.update(loss.data.item(), batch_size)
    
            self.print_log(f'===== Test ====> Prec@1 {top1.avg:.3f}% Prec@5 {top5.avg:.3f}%', log)
        
        return top1.avg

    

# ----------------------------------------------------------------------------------------------------------------------
class Trainer_lite(Trainer_Base):
    """
        1. no tensorboard
        2. no resume
        3. no ddp
    """

    def __init__(self, args):

        # ---
        if args.disable_amp:
            self.scaler = None
        else:
            self.scaler = torch.cuda.amp.GradScaler()

        super().__init__(args)
    
    
    def preprocess_train_sample(self, args, x: torch.Tensor):
        return x

    def preprocess_test_sample(self, args, x: torch.Tensor):
        return x

    def process_model_output(self, args, y: torch.Tensor):
        return y


    @staticmethod
    def prepare_datasets(args):
        
        assert 'train' in os.listdir(args.data_path)     # assume the dataset is organized by ImageNet style
        
        traindir = os.path.join(args.data_path, "train")
        valdir = os.path.join(args.data_path, "val")
        
        val_resize_size, val_crop_size, train_crop_size = args.val_resize_size, args.val_crop_size, args.train_crop_size
        interpolation = InterpolationMode(args.interpolation)

        auto_augment_policy = getattr(args, "auto_augment", None)
        random_erase_prob = getattr(args, "random_erase", 0.0)
        
        dataset_train = torchvision.datasets.ImageFolder(
                                                traindir,
                                                presets.ClassificationPresetTrain(
                                                    crop_size=train_crop_size,
                                                    interpolation=interpolation,
                                                    auto_augment_policy=auto_augment_policy,
                                                    random_erase_prob=random_erase_prob,
                                                    ),
                                                )

        dataset_val = torchvision.datasets.ImageFolder(
                                                        valdir, 
                                                        presets.ClassificationPresetEval(
                                                            crop_size=val_crop_size, 
                                                            resize_size=val_resize_size, 
                                                            interpolation=interpolation
                                                            ),
                                                        )
        
        return dataset_train, dataset_val
    

    def prepare_dataloaders(self, args):
        
        def _seed_worker(worker_id):
        
            worker_seed = torch.initial_seed() % int(np.power(2, 32))

            np.random.seed(worker_seed)
            random.seed(worker_seed)

        loader_g = torch.Generator()
        loader_g.manual_seed(args.seed)
        
        train_sampler = torch.utils.data.RandomSampler(self.dataset, generator=loader_g)
        test_sampler = torch.utils.data.SequentialSampler(self.dataset_val)
        
        collate_fn = None
        self.num_classes = len(self.dataset.classes)
        
        mixup_transforms = []
        if args.mixup_alpha > 0.0:
            mixup_transforms.append(transforms.RandomMixup(self.num_classes, p=1.0, alpha=args.mixup_alpha))
        if args.cutmix_alpha > 0.0:
            mixup_transforms.append(transforms.RandomCutmix(self.num_classes, p=1.0, alpha=args.cutmix_alpha))
        if mixup_transforms:
            mixupcutmix = torchvision.transforms.RandomChoice(mixup_transforms)
            collate_fn = lambda batch: mixupcutmix(*default_collate(batch))  # noqa: E731
            
        self.train_loader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=args.batch_size,
            sampler=train_sampler,
            num_workers=args.workers,
            pin_memory= not args.disable_pinmemory,
            collate_fn=collate_fn,
            worker_init_fn=_seed_worker
        )

        self.val_loader = torch.utils.data.DataLoader(
            self.dataset_val, 
            batch_size=args.batch_size, 
            sampler=test_sampler, 
            num_workers=args.workers, 
            pin_memory= not args.disable_pinmemory,
            worker_init_fn=_seed_worker
        )


    @staticmethod
    def set_optimizer(args, model):
        
        if args.norm_weight_decay is None:
            parameters = model.parameters()
        else:
            param_groups = torchvision.ops._utils.split_normalization_params(model)
            wd_groups = [args.norm_weight_decay, args.weight_decay]
            parameters = [{"params": p, "weight_decay": w} for p, w in zip(param_groups, wd_groups) if p]
        
        optimizer = torch.optim.SGD(
            parameters,
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            nesterov=True,
        )
       
        return optimizer
    
    
    @staticmethod
    def set_lr_scheduler(args, optimizer):

        main_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs - args.lr_warmup_epochs)
        warmup_lr_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=args.lr_warmup_decay, total_iters=args.lr_warmup_epochs)
        lr_scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup_lr_scheduler, main_lr_scheduler], milestones=[args.lr_warmup_epochs])

        return lr_scheduler
    
    
    def train(self, args, verbose=False, **kwargs) -> None:

        # --- prepare for training arguments
        optimizer = self.set_optimizer(args, self.model)
        lr_scheduler = self.set_lr_scheduler(args, optimizer)

        # -----
        current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        for epoch in tqdm(range(args.start_epoch, args.epochs), desc=f'{current_time} Training'):     # for every epoch
            
            train_acc1, train_acc5, train_loss = self.train_one_epoch(optimizer, epoch, args, verbose=verbose)

            lr_scheduler.step()

            if verbose:
                #print(f'escape time={(datetime.datetime.now() + datetime.timedelta(seconds=(time.time() - start_time) * (args.epochs - epoch))).strftime("%Y-%m-%d %H:%M:%S")}\n')
                print(f'Epoch [{epoch}] -> acc@1: {train_acc1:.3f}, acc@5: {train_acc5:.3f}, loss: {train_loss:.5f}')
                print(optimizer.state_dict()['param_groups'][0]['lr'])


    def train_one_epoch(self, args, optimizer, epoch, log):
        
        self.model.train()

        top1 = AverageMeter()
        top5 = AverageMeter()
        _loss = AverageMeter()

        for i, (image, target) in enumerate(self.train_loader):

            image = image.to(self.device, non_blocking=True)
            target = target.to(self.device, non_blocking=True)
            
            with torch.amp.autocast('cuda', enabled=self.scaler is not None):

                image = self.preprocess_train_sample(args, image).to(args.device)
                output = self.process_model_output(args, self.model(image))
                loss = self.criterion(output, target)

            optimizer.zero_grad()
            
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                loss.backward()
                optimizer.step()

            #functional.reset_net(self.model)

            acc1, acc5 = self.accuracy(output, target, topk=(1, 5))
            batch_size = target.shape[0]
            
            top1.update(acc1.item(), batch_size)
            top5.update(acc5.item(), batch_size)
            _loss.update(loss.item(), batch_size)
        
        self.print_log(f'===== [{epoch}/{args.epochs}] Train ====> Acc@1 {top1.avg:.3f}% Acc@5 {top5.avg:.3f}%', log)


    def validate(self, args, log):
        
        self.model.eval()
       
        top1 = AverageMeter()
        top5 = AverageMeter()
        _loss = AverageMeter()
        
        with torch.inference_mode():
            
            for i, (image, target) in enumerate(self.val_loader):
                
                image = image.to(self.device, non_blocking=True)
                target = target.to(self.device, non_blocking=True)
                image = self.preprocess_test_sample(args, image)

                output = self.process_model_output(args, self.model(image))
                loss = self.criterion(output, target)

                acc1, acc5 = self.accuracy(output.data, target, topk=(1, 5))
                batch_size = target.shape[0]

                #functional.reset_net(self.model)
                
                top1.update(acc1.item(), batch_size)
                top5.update(acc5.item(), batch_size)
                _loss.update(loss.item(), batch_size)

        self.print_log(f'===== Test ====> Acc@1 {top1.avg:.3f}% Acc@5 {top5.avg:.3f}%', log)
        
        return top1.avg


# ----------------------------------------------------------------------------------------------------------------------
class AverageMeter():  
    
    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count