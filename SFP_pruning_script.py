#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
General pruning script

@author: acxyle
"""

# --- python
import os
import sys
import torch
import torchvision

# --- local
import SFP_Pruning
import Trainer

import models
import utils


# ======================================================================================================================
model_names = sorted(name for name in models.__dict__ if name.islower() and not name.startswith("__") and callable(models.__dict__[name]))

# ----------------------------------------------------------------------------------------------------------------------
def get_args_parser(add_help=True):
    """ 
    this section only determine the shared arguments, see Trainer.training_parser() for training parsers and 
    SFP_Pruning.SFP_pruning_parser() for pruning arguments
    """
    
    parser = Trainer.training_parser()
    parser = SFP_Pruning.SFP_pruning_parser(parser)
    
    # ---
    parser.add_argument("--num_classes", type=int, default=2622)

    # --- env config
    parser.add_argument("--data_dir", type=str, default='/home/acxyle-workstation/Dataset')
    parser.add_argument("--dataset", type=str, default='CelebA_fold_0')

    parser.add_argument('--arch', default='resnet18', choices=model_names)
    parser.add_argument('--epochs', default=300, type=int, metavar='N', help='number of total epochs to run')

    parser.add_argument('-b', '--batch-size', default=64, type=int, metavar='N', help='mini-batch size')
    parser.add_argument('--lr', default=0.1, type=float, metavar='LR', help='initial learning rate')
    
    # ---
    parser.add_argument("--mode", default="AFPGM", choices=["SFP", "ASFP", "FPGM", "AFPGM"])

    args = parser.parse_args()
    args.data_path = os.path.join(args.data_dir, args.dataset)
    args.log_postfix = f"{args.arch}_{args.dataset}"
    args.output_dir = os.path.join(f"./logs/SFP_Pruning_{args.mode}_{args.log_postfix}_e{args.epochs}_p{args.prune_rate}")
    
    args.timestamp = utils.time_file_str()

    return args


# ----------------------------------------------------------------------------------------------------------------------
class SFP_Pruner:
    """
        Pruner is the pipeline, Trainer and Mask operate the same model
    """
    
    def __init__(self, args):
        
        self.trainer = SFP_Trainer_lite(args)
        
        if args.mode == 'SFP':
            self.mask = SFP_Pruning.SFP_Mask(args, self.model)
        elif args.mode == 'FPGM':
            self.mask = SFP_Pruning.FPGM_Mask(args, self.model)
        elif args.mode == 'ASFP':
            self.mask = SFP_Pruning.ASFP_Mask(args, self.model)
        elif args.mode == 'AFPGM':
            self.mask = SFP_Pruning.AFPGM_Mask(args, self.model)
        else:
            raise ValueError
        
    
    @property
    def model(self):
        
        return self.trainer.model
        
    
    def __call__(self, args):     # --- this function should contains the __main__() for pruning-when_training
        
        os.makedirs(args.output_dir, exist_ok=True)
            
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
        self.train(log)
        
        log.close()
    
    
    def train(self, log, **kwargs):     # --- the train loop is defined outside for the pruning-when-training
        
        self.trainer.best_pred_acc = -1
    
        optimizer = self.trainer.set_optimizer(args, self.model)
        lr_scheduler = self.trainer.set_lr_scheduler(args, optimizer)
    
        for epoch in range(args.epochs):     
            
            self.trainer.print_log("===== LR ====> {}".format(optimizer.param_groups[0]['lr']), log)
            
            # --- train
            self.trainer.train_one_epoch(args, optimizer, epoch, log)
            
            lr_scheduler.step()
            
            # --- pruning
            if (epoch % args.epoch_prune == 0 or epoch == args.epochs - 1):
                
                self.mask.reset(self.model)
                self.mask.mask_initialization(args, epoch, **kwargs)
                self.mask.mask_apply(**kwargs)
                self.mask.count_zero_weights(args)
            
            # --- val after pruning
            val_acc = self.trainer.validate(args, log)

            self.trainer.save_checkpoint(args, val_acc, epoch, optimizer)
            
            
# ----------------------------------------------------------------------------------------------------------------------
class SFP_Trainer_basic(Trainer.Trainer_basic):     # --- for simple dataset
    
    def __init__(self, args):
        
        super().__init__(args)
    
    
    def load_model(self, args):     # --- current design has no entrance to load model

        self.model = models.__dict__[args.arch](num_classes=args.num_classes)


# ----------------------------------------------------------------------------------------------------------------------
class SFP_Trainer_lite(Trainer.Trainer_lite):
    
    def __init__(self, args):
        
        super().__init__(args)
    
    
    def load_model(self, args):

        self.model = models.__dict__[args.arch](num_classes=args.num_classes)


       
# ======================================================================================================================
if __name__ == "__main__":
    
    args = get_args_parser()
    
    # -----
    pruner = SFP_Pruner(args)
        
    pruner(args)