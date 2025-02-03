#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
General pruning script

@author: acxyle

TODO:
    1. add resume
    2. add tf
    3. add parallel
    
"""

# --- python
import os
import sys
import torch
import torchvision
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from typing import Union

from joblib import Parallel, delayed

# --- local
import models
import pruning
import training
import utils


# ======================================================================================================================
model_names = sorted(name for name in models.__dict__ if name.islower() and not name.startswith("__") and callable(models.__dict__[name]))

# ----------------------------------------------------------------------------------------------------------------------
def get_args_parser(add_help=True):
    """ 
        this section only determines the shared arguments, see Trainer.training_parser() for training arguments and 
        SFP_Pruning.SFP_pruning_parser() for pruning arguments
    """
    
    parser = training.training_parser()
    parser = pruning.SFP_pruning_parser(parser)
    
    # ---
    parser.add_argument("--num_classes", type=int, default=2622)

    # --- env config
    parser.add_argument("--data_dir", type=str, default='/home/acxyle-workstation/Dataset')
    parser.add_argument("--dataset", type=str, default='C2k')
    parser.add_argument("--kfold_training", type=bool, default=True)
    parser.add_argument("--kfold_number", type=int, default=5)
    parser.add_argument("--kfold_idces", type=list, default=[1,3])     # --- if None, train all folds
    parser.add_argument("--kfold_training_parallel", type=bool, default=False)

    parser.add_argument('--arch', default='vgg16_bn', choices=model_names)
    parser.add_argument('--epochs', default=300, type=int, metavar='N', help='number of total epochs to run')

    parser.add_argument('-b', '--batch-size', default=32, type=int, metavar='N', help='mini-batch size')
    parser.add_argument('--lr', default=0.01, type=float, metavar='LR', help='initial learning rate')
    
    # ---
    parser.add_argument("--mode", default="AFPGM", choices=["SFP", "ASFP", "FPGM", "AFPGM"])
    parser.add_argument("--prune_classifier", type=bool, default=True)     # --- only applicable for classifier with multiple linear layers
    
    args = parser.parse_args()
    args.data_path = os.path.join(args.data_dir, args.dataset)
    args.log_postfix = f"{args.arch}_{args.dataset}"

    args.output_dir = os.path.join(f"./logs/SFP_Pruning_{args.mode}_{args.log_postfix}_e{args.epochs}_pn{args.prune_rate}")

    if args.mode in ['FPGM', 'AFPGM']:
        args.output_dir = ''.join([args.output_dir, f"_pd{args.prune_rate_distance}"])
    
    args.timestamp = utils.time_file_str()

    return args



def skfold_training_script(args):
    """
        write for loop first and convert to joblib
    """
    

    skfold_indices = np.arange(args.kfold_number)
    assert all([(_ in skfold_indices) for _ in args.kfold_idces])
    
    for skfold_idx in args.kfold_idces:
    
        pruner = SFP_Pruner(args, kfold_idx=skfold_idx)
            
        pruner(args)


# ----------------------------------------------------------------------------------------------------------------------
class SFP_Pruner:
    """
        Pruner is the pipeline, Trainer and Mask operate the same model
    """
    
    def __init__(self, args, kfold_idx=0, **kwargs):
        
        if args.kfold_training:
            args.command = 'skfold'
            args.kfold_idx = kfold_idx
        
        self.trainer = SFP_Trainer_lite(args, **kwargs)
        
        if args.mode == 'SFP':
            self.mask = pruning.SFP_Mask(args, self.model)
        elif args.mode == 'FPGM':
            self.mask = pruning.FPGM_Mask(args, self.model)
        elif args.mode == 'ASFP':
            self.mask = pruning.ASFP_Mask(args, self.model)
        elif args.mode == 'AFPGM':
            self.mask = pruning.AFPGM_Mask(args, self.model)
        else:
            raise ValueError

        # ---
        args.output_dir = os.path.join(args.output_dir, f'runs/{args.kfold_idx}')
        os.makedirs(args.output_dir, exist_ok=True)

        # ---
        self.writer = SummaryWriter(log_dir=os.path.join(args.output_dir, 'tb'))
        
    
    @property
    def model(self):
        
        return self.trainer.model
        
    
    def __call__(self, args):     # --- this function should contains the __main__() for pruning-when_training
         
        log = open(os.path.join(args.output_dir, f'{args.mode}-{args.arch}-{args.timestamp}.log'), 'w')
        
        self.trainer.print_log(f"===== Config ====> {args}", log)
        
        # version information
        self.trainer.print_log("===== Python Version ====> {}".format(sys.version.replace('\n', ' ')), log)
        self.trainer.print_log("===== Pytorch Version ====> {}".format(torch.__version__), log)
        self.trainer.print_log("===== cuDNN Version =====> {}".format(torch.backends.cudnn.version()), log)
        self.trainer.print_log("===== Vision Version ====> {}".format(torchvision.__version__), log)
        
        # create model
        self.trainer.print_log(f"===== Model ====>\n {self.trainer.model}", log)
        
        # --- pruning setting
        self.mask.mask_initialization(args)
        self.mask.mask_apply()
       
        # --- val test
        #self.trainer.validate(args, log)
        
        # ---
        self.train(args, log)
        
        log.close()
        self.writer.close()
    
    
    def train(self, args, log, **kwargs):     # --- the train loop is defined outside for the pruning-when-training
        
        # --- tf recorder
        images, labels = next(iter(self.trainer.train_loader))
        grid = torchvision.utils.make_grid(images)
        self.writer.add_image('images', grid, 0)

        dummy_input = torch.zeros_like(images[0, ...]).to(args.device)
        self.writer.add_graph(self.model, dummy_input.unsqueeze(0))

        # ---
        def report_prune_rate(dynamic_pruning_rate, dynamic_pruning_rate_distance=None):
            
            text_pruning_rate = f"===== [{epoch}/{args.epochs}] prune_rate ====> {dynamic_pruning_rate*100:.2f}%"
            self.trainer.print_log(text_pruning_rate, log)
            self.writer.add_scalar('prune_rate_norm', dynamic_pruning_rate, epoch)
            
            if dynamic_pruning_rate_distance is not None:
            
                text_pruning_rate_distance = f"===== [{epoch}/{args.epochs}] prune_rate_distance ====> {dynamic_pruning_rate_distance*100:.2f}%"
                self.trainer.print_log(text_pruning_rate_distance, log)
                self.writer.add_scalar('prune_rate_distance', dynamic_pruning_rate_distance, epoch)
    
        self.trainer.best_pred_acc = -1
        
        optimizer = self.trainer.set_optimizer(args, self.model)
        lr_scheduler = self.trainer.set_lr_scheduler(args, optimizer)
    
        for epoch in range(args.epochs):     
            
            self.trainer.print_log("===== LR ====> {}".format(optimizer.param_groups[0]['lr']), log)
            
            # --- train
            acc1_train, acc5_train, loss_train = self.trainer.train_one_epoch(args, optimizer, epoch, log)
            
            lr_scheduler.step()
            
            # --- pruning
            if (epoch % args.epoch_prune == 0 or epoch == args.epochs - 1):
                
                self.mask.reset(self.model)
                
                if args.mode == 'AFPGM':
                    dynamic_pruning_rate, dynamic_pruning_rate_distance = self.mask.mask_initialization(args, epoch=epoch, **kwargs)
                    report_prune_rate(dynamic_pruning_rate, dynamic_pruning_rate_distance)
                elif args.mode == 'ASFP':
                    dynamic_pruning_rate = self.mask.mask_initialization(args, epoch=epoch, **kwargs)
                    report_prune_rate(dynamic_pruning_rate)
                
                self.mask.mask_apply(**kwargs)
                self.mask.count_zero_weights(**kwargs)
            
            # --- val after pruning
            acc1_val, acc5_val, loss_val = self.trainer.validate(args, log)

            self.trainer.save_checkpoint(args, acc1_val, epoch, optimizer)

            # --- (3) record loss
            self.writer.add_scalars('loss', {'train': loss_train, 'val': loss_val}, epoch)

            # --- (4) record acc
            self.writer.add_scalars('acc@1', {'train': acc1_train, 'val': acc1_val}, epoch)
            self.writer.add_scalars('acc@5', {'train': acc5_train, 'val': acc5_val}, epoch)

            # --- (5) record lr, prune_rate
            self.writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)

            
      
# ----------------------------------------------------------------------------------------------------------------------
class SFP_Trainer_basic(training.Trainer_basic):     # --- for small-scale data like mnist and cifar
    
    def __init__(self, args):
        
        super().__init__(args)
    
    def load_model(self, args):     # --- current design has no entrance to load model

        self.model = models.__dict__[args.arch](num_classes=args.num_classes)


# ----------------------------------------------------------------------------------------------------------------------
class SFP_Trainer_lite(training.Trainer_lite):     # --- general natural image
    
    def __init__(self, args, **kwargs):

        super().__init__(args, **kwargs)
    
    def load_model(self, args):

        self.model = models.__dict__[args.arch](num_classes=args.num_classes)


       
# ======================================================================================================================
if __name__ == "__main__":
    
    args = get_args_parser()
    
    skfold_training_script(args)