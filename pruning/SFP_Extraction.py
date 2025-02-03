#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extract the pruned subnetwork. The only criteria is the model weight, the kernel will be removed if sum(kernel) == 0.

this method inherits from 
    [Unofficial PyTorch implementation of pruning VGG on CIFAR-10 Data set]
    @author: tyui592
    https://github.com/tyui592/Pruning_filters_for_efficient_convnets

the basic functions remain the same design, but the overall structure has been significantly changed, and extraction only

---
    **SFP**: Soft Filter Pruning;
    **CP**: Convnet Pruning, or, **HFP**: Hard Filter Pruning

@author: acxyle
"""


import os
import copy
from tqdm import tqdm
import numpy as np

from typing import Union, List

import torch
import torch.nn as nn
import torchvision
from torchvision.models.resnet import BasicBlock, Bottleneck

from . import SFP_Extraction_utils


# ----------------------------------------------------------------------------------------------------------------------
class CP_Extractor_Base():

    def __init__(self, ):
        
        ...

    def _network_update(self, args, top1):
        
        if top1 > (self.best_acc-self.tolerance):
        
            self.best_acc = top1
            self.best_network = copy.deepcopy(self.network)
            
        self.network = copy.deepcopy(self.best_network)
        self.cp_trainer.model = self.network

        if args.save:
            save_path = os.path.join(args.output_dir, f'{args.model}_checkpoint_max_test_acc1.pth')
            torch.save(self.network, save_path)
     
        
    @staticmethod
    def Prune_Block_Conv2d(target_block, conv_name, dim, channel_index, frame='torch', verbose=False, **kwargs) -> None:
    
        assert channel_index is not None, 'channel_index can not be None'
    
        target_conv = getattr(target_block, conv_name)
        new_conv = SFP_Extraction_utils.get_new_conv(target_conv, dim, channel_index, frame=frame, **kwargs)
    
        if verbose:
            print(conv_name, target_conv.weight.data.shape, '->', new_conv.weight.data.shape)
    
        setattr(target_block, conv_name, new_conv)     # --- 'new_conv' replace 'conv' by name in target_block
    
    
    @staticmethod
    def Prune_Block_BatchNorm2d(target_block, bn_name, channel_index, frame='torch', **kwargs) -> None:
    
        bn_module = getattr(target_block, bn_name)
        new_bn = SFP_Extraction_utils.get_new_norm(bn_module, channel_index, frame=frame, **kwargs)
        setattr(target_block, bn_name, new_bn)
    
    
    @staticmethod
    def Prune_Block_Linear(target_block, fc_name, dim, channel_index, frame='torch', verbose=False, **kwargs) -> None:
        
        assert channel_index is not None, 'channel_index can not be None'
    
        target_fc = getattr(target_block, fc_name)
        new_fc = SFP_Extraction_utils.get_new_linear(target_fc, dim, channel_index, frame=frame, **kwargs)
    
        if verbose:
            print(fc_name, target_fc.weight.data.shape, '->', new_fc.weight.data.shape)
    
        setattr(target_block, fc_name, new_fc)    



class CP_Extractor_VGG(CP_Extractor_Base):
    
    def __init__(self, **kwargs) -> None:
        
        super().__init__(**kwargs)

        self.prune_blocks = ['features', 'classifier']
        
        
    def __call__(self, model, verbose=True, **kwargs) -> nn.Module:
        
        if verbose:
            
            print(f'Pruning Blocks: {self.prune_blocks}')

        self.network = model        

        # ---
        for _block in self.prune_blocks:     
            
            if _block == 'features':
            
                self.pruning_block(_block)
            
            elif _block == 'classifier':
                
                self.pruning_classifier(_block)
            
            else:
                
                raise AssertionError
        
        return self.network

    
    def _network_decomposition(self, prune_block:str) -> None:
        self.target_block = getattr(self.network, prune_block)
            
            
    def pruning_block(self, prune_block:str, **kwargs) -> nn.Module:
        """
           ...
        """
         
        # --- init
        self.prune_block = prune_block
        self._network_decomposition(prune_block)
        
        # ---
        self.block_convs_list = [_conv for _conv, _ in self.target_block.named_modules() if isinstance(_, nn.Conv2d)]
        self.end_conv_layer = self.block_convs_list[-1]
        
        # ---
        for idx, _conv in tqdm(enumerate(self.block_convs_list), desc='Pruning Conv', total=len(self.block_convs_list)): 
           
            self.pruning_conv(_conv, device='cuda:0')

                                
    def pruning_conv(self, conv_name:str, **kwargs) -> nn.Module:
        """
            ...   
        """
        
        conv = getattr(self.target_block, conv_name)
        (out_channels, in_channels, _, _) = conv.weight.shape
        
        # --- return the non-zero kernel indices
        channel_index = SFP_Extraction_utils.get_channel_index(conv.weight.data)
        
        # --- operate with the target conv layer
        self.Prune_Block_Conv2d(self.target_block, conv_name, 0, channel_index, **kwargs)
        
        if isinstance(getattr(self.target_block, str(int(conv_name)+1)), nn.BatchNorm2d):
            self.Prune_Block_BatchNorm2d(self.target_block, f'{int(conv_name)+1}', channel_index, **kwargs)

        # --- operate with the in_channel of the following layer - also in this block
        conv_idx = self.block_convs_list.index(conv_name)
        
        if conv_name != self.end_conv_layer:
            self.Prune_Block_Conv2d(self.target_block, self.block_convs_list[conv_idx+1], 1, channel_index, **kwargs)
            
        else:
            self.network.classifier[0] = SFP_Extraction_utils.get_new_linear(self.network.classifier[0], 1, channel_index, self.network.avgpool, **kwargs)
        
        # ---
        channel_index = None
        
    
    def pruning_classifier(self, prune_block:str, **kwargs) -> nn.Module:
        
        # --- init
        self.prune_block = prune_block
        self._network_decomposition(prune_block)
        
        # --- exclude the final fc layer
        self.block_fcs_list = [_fc for _fc, _ in self.target_block.named_modules() if isinstance(_, nn.Linear)]
        self.end_fc_layer = self.block_fcs_list[-1]

        # ---
        for idx, _fc in tqdm(enumerate(self.block_fcs_list[:-1]), desc='Pruning Linear', total=len(self.block_fcs_list[:-1])):
            
            self.pruning_linear(_fc, device='cuda:0')
                

    
    def pruning_linear(self, linear_name:str, **kwargs) -> nn.Module:
        """
            ...
        """
        
        linear = getattr(self.target_block, linear_name)
        (out_features, in_features) = linear.weight.shape
        
        # --- obtain the unwanted channel indices
        channel_index = SFP_Extraction_utils.get_channel_index(linear.weight.data)
        
        # --- operate with the target conv layer
        self.Prune_Block_Linear(self.target_block, linear_name, 0, channel_index, **kwargs)
        
        # --- operate with BN, while standard VGG does not have it
        if hasattr(self.target_block, next_layer_name:=str(int(linear_name)+1)):
            if isinstance(getattr(self.target_block, next_layer_name), nn.BatchNorm2d):
                self.Prune_Block_BatchNorm2d(self.target_block, f'{int(linear_name)+1}', channel_index, **kwargs)

        # --- operate with the following layer - also in this block
        fc_idx = self.block_fcs_list.index(linear_name)
        
        self.Prune_Block_Linear(self.target_block, self.block_fcs_list[fc_idx+1], 1, channel_index, **kwargs)
            
        # ---
        channel_index = None
        
        

class CP_Extractor_ResNet(CP_Extractor_Base):
    
    def __init__(self, **kwargs):
        
        # --- init
        super().__init__(**kwargs) 
        
        # ---
        
    def __call__(self, model, exclude_final_conv=True, **kwargs):
        """
            exclude_final_conv: if False, prune the entire model, this will compromise the computation process due to the
            SFP pruning actually prunes the module independetly, thus the pruned channels in residual path and shortcut path
            are different. The function is kept for future use when adjust the SFP training as well, i.e., prune the 
            same channels during pruning-at-training process of multi-pathway models like Resnet.
        """
        
        self.network = model
        
        # --- decide to prune the first conv
        if not exclude_final_conv:
            
            self.pruning_stem()
        
        # --- prune the main architecture
        self.prune_blocks = [_n for _n,_m in model.named_modules() if isinstance(_m, (BasicBlock, Bottleneck))]
        
        for _block in self.prune_blocks:     
            
            self.pruning_block(_block, exclude_final_conv)
        
        return self.network
               
    
    def pruning_stem(self, conv_name='conv1', device='cuda:0', **kwargs) -> nn.Module:
        """
            pending...
            
            put the adjust of channel_in here if next block has downsample
        """
        
        # --- prune the first conv
        conv = getattr(self.network, conv_name)
        channel_index = SFP_Extraction_utils.get_channel_index(conv.weight.data)
        
        self.Prune_Block_Conv2d(self.network, conv_name, 0, channel_index, device=device, **kwargs)
        self.Prune_Block_BatchNorm2d(self.network, f'bn{conv_name[-1]}', channel_index, device=device, **kwargs)
        
        # --- adjust the channel_in of te subsequent conv
        target_stage_next = getattr(self.network, 'layer1')
        self.Prune_Block_Conv2d(target_stage_next[0], 'conv1', 1, channel_index, device=device, **kwargs)
        
        if hasattr(target_stage_next[0], 'downsample'):
            
            self.Prune_Block_Conv2d(target_stage_next[0].downsample, '0', 1, channel_index, device=device, **kwargs)
        

    
    def pruning_block(self, prune_block:str, exclude_final_conv:bool=True, **kwargs) -> nn.Module:
        """
           ...
        """
         
        # --- init
        self.prune_block = prune_block
        (self.prune_block_l, self.prune_block_b) = prune_block.split('.')
        
        frame = 'torch'
        
        def _network_decomposition():
            # --- this extract the target block from the model
            self.target_stage = getattr(self.network, self.prune_block_l)     # --- stage, nn.Module
            self.target_block = getattr(self.target_stage, self.prune_block_b)    # --- block, nn.Module
            
            self.prune_stage_l_idx = int(self.prune_block_l[-1])     # --- idx of current stage
            self.prune_block_b_idx = int(self.prune_block_b)     # --- idx of current block
            
            # --- this changes along stage or conv
            self.network_stages_list: List[int] = [int(_name[-1]) for _name, _ in self.network.named_children() if 'layer' in _name]
            self.stage_blocks_list: List[int] = [int(_name) for _name, _ in self.target_stage.named_children()]
        
        # ---
        _network_decomposition()
        
        self.block_convs_list = [_conv for _conv, _ in self.target_block.named_modules() if isinstance(_, nn.Conv2d) and ('downsample' not in _conv)]
        self.end_conv_layer = self.block_convs_list[-1]
        
        # ---
        _module = getattr(self.target_block, 'conv1')
        (out_channels, _, _, _) = _module.weight.shape
        
        for _conv in self.block_convs_list:
            
            if exclude_final_conv:
                self.pruning_conv(_conv, frame=frame, **kwargs)     # -> self.network
            else:
                self.pruning_conv_all(_conv, frame=frame, **kwargs)
                
            
    def pruning_conv(self, conv_name:str, device:str='cuda:0', **kwargs) -> None:
        """
            this function does not prune the last conv of residual pathway and the downsample conv of shortcut pathway    
        """
        conv = getattr(self.target_block, conv_name)
        (out_channels, in_channels, _, _) = conv.weight.shape
        
        # ---
        if conv_name != self.end_conv_layer:      # --- if the layer is not the last conv
            
            # --- 
            channel_index = SFP_Extraction_utils.get_channel_index(conv.weight.data)

            # --- operate with the target conv
            self.Prune_Block_Conv2d(self.target_block, conv_name, 0, channel_index, device=device, **kwargs)
            self.Prune_Block_BatchNorm2d(self.target_block, f'bn{conv_name[-1]}', channel_index, device=device, **kwargs)

            # --- operate with the following conv - also in this block
            self.Prune_Block_Conv2d(self.target_block, f'conv{int(conv_name[-1])+1}', 1, channel_index, device=device, **kwargs)
            
        else:     # --- if the layer is the last one
        
            ...

        self.network.to(device)
        
        
    def pruning_conv_all(self, conv_name:str, device:str='cuda:0', **kwargs) -> nn.Module:
        """
            this function prunes all conv layers
            TODO: adjust the channel multi-use of resnet if the model has been pruned by shared channels
        """
        
        conv = getattr(self.target_block, conv_name)
        (out_channels, in_channels, _, _) = conv.weight.shape
        
        # --- operate with the target conv
        channel_index = SFP_Extraction_utils.get_channel_index(conv.weight.data)

        self.Prune_Block_Conv2d(self.target_block, conv_name, 0, channel_index, device=device, **kwargs)
        self.Prune_Block_BatchNorm2d(self.target_block, f'bn{conv_name[-1]}', channel_index, device=device, **kwargs)

        # ---
        if conv_name != self.end_conv_layer:
            
            # --- operate with the following conv - also in this block
            self.Prune_Block_Conv2d(self.target_block, f'conv{int(conv_name[-1])+1}', 1, channel_index, device=device, **kwargs)
        
        else:
            
            if getattr(self.target_block, 'downsample') is None:
     
                if self.prune_block_b_idx+1 in self.stage_blocks_list:     # if the next BLOCK exists, need to replace the 1st conv and the shortcut the next BLOCK

                    target_block_next = getattr(self.target_stage, f'{self.prune_block_b_idx+1}')

                    self.Prune_Block_Conv2d(target_block_next, 'conv1', 1, channel_index, device=device, **kwargs)

                elif self.prune_stage_l_idx+1 in self.network_stages_list:     # if the next BLOCK does NOT exists and this network has next STAGE
                    
                    target_stage_next = getattr(self.network, f'layer{self.prune_stage_l_idx+1}')
                    target_block_next = getattr(target_stage_next, '0')

                    self.Prune_Block_Conv2d(target_block_next, 'conv1', 1, channel_index, device=device, **kwargs)
                    self.Prune_Block_Conv2d(target_block_next.downsample, '0', 1, channel_index, device=device, **kwargs)

            else:
                
                channel_index = SFP_Extraction_utils.get_channel_index(self.target_block.downsample[0].weight.data)
                
                # --- 
                self.Prune_Block_Conv2d(self.target_block.downsample, '0', 0, channel_index, device=device, **kwargs)
                self.Prune_Block_BatchNorm2d(self.target_block.downsample, '1', channel_index, device=device, **kwargs)
                
                # --- 
                target_block_next = getattr(self.target_stage, f'{self.prune_block_b_idx+1}')
    
                self.Prune_Block_Conv2d(target_block_next, 'conv1', 1, channel_index, device=device, **kwargs)
                
                # ---
                if self.prune_stage_l_idx+1 not in self.network_stages_list: 
                    
                    self.network.fc = SFP_Extraction_utils.get_new_linear(self.network.fc, 1, channel_index, device=device, **kwargs)

   
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    
    model = torchvision.models.vgg16_bn(num_classes=2622)
    model = model.to('cuda:0')
    params_path = '/home/acxyle-workstation/Github/SFP [local]/logs/SFP_Pruning_AFPGM_vgg16_bn_C2k_e300_pn0.4_pd0.5/runs/0/best-vgg16_bn-2025-02-02-8837.pth.tar'
    params = torch.load(params_path, weights_only=False)
    params = params['state_dict']
    model.load_state_dict(params)    
    
    Pruner = CP_Extractor_VGG()
    downsized_model = Pruner(model)
    
# =============================================================================
#     model = torchvision.models.resnet50(num_classes=2622)
#     model = model.to('cuda:0')
#     params_path = '/home/acxyle-workstation/Github/SFP [local]/logs/Resnet/SFP_Pruning_FPGM_resnet50_C2k_e300_pn0.4_pd0.5/best-resnet50-2025-01-30-5621.pth.tar'
#     params = torch.load(params_path, weights_only=False)
#     params = params['state_dict']
#     model.load_state_dict(params)    
#     
#     Pruner = CP_Extractor_ResNet()
#     downsized_model = Pruner(model)
# =============================================================================
    
    torch.save(downsized_model, os.path.join('/'.join(params_path.split('/')[:-1]), 'downsized_model.pth'))