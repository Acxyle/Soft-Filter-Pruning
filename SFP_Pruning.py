#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyTorch implementation for 

    "Soft Filter Pruning for Accelerating Deep Convolutional Neural Networks" IJCAI 2018
        refer to: https://github.com/he-y/soft-filter-pruning
            
    "Filter Pruning via Geometric Median for Deep Convolutional Neural Networks Acceleration" CVPR Oral 2019
        refer to: https://github.com/he-y/filter-pruning-geometric-median
            
    "Asymptotic Soft Filter Pruning for Deep Convolutional Neural Networks" IEEE Trans Cybernetics 2020
        no publicly available original code

@author: HE Yang
@modified: acxyle

this code only provide Masks, no training parts.
only default configs has been tested, some parameters may not work.

"""

import math
import numpy as np
from typing import Any

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data

# ======================================================================================================================
def SFP_pruning_parser(parser):

    parser.add_argument('--prune_rate', type=float, default=0.3)
    parser.add_argument('--prune_rate_distance', type=float, default=0.4)
    
    parser.add_argument('--epoch_prune', type=int, default=1, help='compress layer of model')
    parser.add_argument('--skip_downsample', type=bool, default=False, help='skip the shortcut connection')
    
    return parser


# ----------------------------------------------------------------------------------------------------------------------
class SFP_Mask_Base:
    """
        Basic functions of Soft Filter Pruning
        
        in the original code of SFP, only (1) weight value has been removed. And in the original code of 
        FPGM, the (1) weight value and (2) weight grad have been removed. 
        
        so the interesting question is how 'clean' of the pruning
        
    """
    
    def __init__(self, model):
        
        self.model = model
        
        self.model_size = {}
        self.prune_rate = {}
        self.mask_dict = {}
        self.mask_index = []
        
        # ---
        for index, (name, module) in enumerate(self.model.named_modules()):
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.Linear):
                self.model_size[index] = module.weight.shape     # --- if the module has bias, it has the same shape
                
    
    def reset(self, model):
        
        SFP_Mask_Base.__init__(self, model)
        
    
    def mask_apply(self, remove_grad=True, **kwargs) -> None:
        """ apply the mask to the weight value """
        
        for index, (name, module) in enumerate(self.model.named_modules()):     # --- for each layer
        
            if index in self.mask_index:
                
                mask = self.mask_dict[index]
                
                module.weight.data = module.weight.data * mask
                
                if hasattr(module, 'bias') and (getattr(module, 'bias') is not None):
                    
                    module.bias.data = module.bias.data * mask
                
                if remove_grad and (module.weight.grad is not None):
                    
                    module.weight.grad.data = module.weight.grad.data * mask
                    
                    if hasattr(module, 'bias') and (getattr(module, 'bias') is not None):
                        
                        module.bias.grad.data = module.bias.grad.data * mask
    
        
    def mask_initialization(self, args, **kwargs) -> None:
        """ current design uses the channel selected from weight of conv """
        
        self.set_pruning_layer_and_ratio(args)
        
        self.mask_generation(args)
    
    
    def mask_generation(self, args, remove_bn=True) -> None:
        
        for index, (name, module) in enumerate(self.model.named_modules()):     # --- for each layer
        
            if index in self.mask_index:
                
                if isinstance(module, nn.Conv2d):
                
                    self.mask_dict[index], filter_index = self.get_conv_mask_by_norm(module.weight.data, index, return_idx=True)
            
                if remove_bn and isinstance(module, nn.BatchNorm2d) and (filter_index is not None):

                    self.mask_dict[index] = self.get_bn_mask(index, filter_index)
    
    
    def set_pruning_layer_and_ratio(self, args) -> None:
        """ this function determines the index of target layers and pruning ratio of each layer """
        
        if 'vgg' in args.arch:
            
            for index, (name, module) in enumerate(self.model.named_modules()):
                
                if isinstance(module, nn.Conv2d) or isinstance(module, nn.BatchNorm2d):
                    
                    self.prune_rate[index] = args.prune_rate
                    self.mask_index.append(index)
                                
        elif "resnet" in args.arch:
            
            for index, (name, module) in enumerate(self.model.named_modules()):

                if isinstance(module, nn.Conv2d) or isinstance(module, nn.BatchNorm2d):
                    
                    self.mask_index.append(index)
                    self.prune_rate[index] = args.prune_rate
                
            # ---
            if args.skip_downsample:      # skip downsample layer
            
                skip_list = []
                for idx, (name, module) in enumerate(self.model.named_modules()):
                    if ('downsample') in name and (isinstance(module, nn.Conv2d) or isinstance(module, nn.BatchNorm2d)):
                        skip_list.append(idx)
            
                for x in skip_list:
                    
                    self.prune_rate.pop(x)
                    self.mask_index.remove(x)
                    
        else:     # --- the channel selection is independent for shortcut-pathway and residual-pathway
            
            raise ValueError
                    
            
    def get_conv_mask_by_norm(self, input, index, return_idx=False) -> torch.Tensor:
        
        filter_index = self.get_filter_indices_by_norm(input, index)
        
        mask = self.get_mask(input, filter_index)
        
        if return_idx:
            
            return mask, filter_index
        
        else:
            
            return mask
                
        
    def get_bn_mask(self, layer_index, filter_index, device='cuda:0') -> Any:
        
        dummy_input = torch.ones(self.model_size[layer_index]).to(device)
        
        return self.get_mask(dummy_input, filter_index)
    
    
    @staticmethod
    def get_mask(input, filter_index) -> torch.Tensor:
        """ ... """
        
        mask = torch.ones_like(input)
        
        mask[filter_index] = 0
        
        return mask
      
    
    def get_filter_indices_by_norm(self, input, index) -> torch.Tensor:
        """
            input: tensor, weight of kernel
            index: layer index
        """
        
        assert input.ndim == 4
        
        filter_pruned_num = int(input.shape[0] * self.prune_rate[index])
        norm = torch.norm(input, p=2, dim=(1,2,3))

        return torch.argsort(norm)[:filter_pruned_num]

    
    def count_zero_weights(self, args) -> None:
        """ this function report the zero and nonzero weight values """
        
        for idx, (name, module) in enumerate(self.model.named_modules()):
           
            if idx in self.mask_dict:
                
                b = getattr(module, 'weight').data.view(self.model_size[idx].numel()).cpu().numpy()
                cnz = np.count_nonzero(b)
                cz = len(b)-np.count_nonzero(b)
                pct_zero = cz/len(b)

                print(f"layer: {idx}, pct_zero is {pct_zero*100:.2f} %, number of nonzero weight is {cnz}, zero is {cz}")
    
    
    @staticmethod
    def dynamic_pruning_rate_calculation(epochs, prune_rate, x3_percentile=0.125, y3_percentile=0.75) -> np.ndarray:
        """ 
            x3_percentile is D in the ASFP paper, and y3_percentile is corresponding hyper param assigned by the authors,
            according to the formula in the paper, y = a * exp(kx) + b -> y = a - a exp(kx), by approximation, a = x3,
            thus, the high order exponential equation downgrades to a logarithmic equation that requires solving for k
            
            ** I'm not so sure is this the author's original design, but it works for my experiments
            
            ** intuitively, the dynamical rates can be calculated by other formula like linear or annealing?
        """

        x3 = epochs * x3_percentile
        y3 = prune_rate * y3_percentile
        
        a = prune_rate
        k = math.log(1-y3/a)/x3
        
        dummy_x = np.arange(epochs)
        dynamic_pruning_rate = a * (1 - np.exp(k * dummy_x))
        
        return dynamic_pruning_rate
                
                
                
class SFP_Mask(SFP_Mask_Base):
    """
        in this code, by default, the (1) weight value, (2) weight grad, and (3) BN have been removed
        if want to remove the impact of Optimizer, remove (4) momentum and (5) other accumulated items
        if want to remove the impact of Other modules with memory, operate all the involved parameters
        
    """
    
    def __init__(self, args, model) -> None:
        
        super().__init__(model)
    
    
    
class ASFP_Mask(SFP_Mask_Base):
    
    def __init__(self, args, model) -> None:
        
        super().__init__(model)
        
        self.dynamic_pruning_rate = self.dynamic_pruning_rate_calculation(args.epochs, args.prune_rate)
    
    
    def mask_initialization(self, args, epoch=0, **kwargs) -> None:
        """ current design uses the channel selected from weight of conv """
        
        self.set_pruning_layer_and_ratio(args, self.dynamic_pruning_rate[epoch])
        
        self.mask_generation(args)
    
    
    def mask_generation(self, args, remove_bn=True) -> None:
        
        for index, (name, module) in enumerate(self.model.named_modules()):     # --- for each layer
        
            if index in self.mask_index:
                
                if isinstance(module, nn.Conv2d):
                
                    self.mask_dict[index], filter_index = self.get_conv_mask_by_norm(module.weight.data, index, return_idx=True)
            
                if remove_bn and isinstance(module, nn.BatchNorm2d) and (filter_index is not None):

                    self.mask_dict[index] = self.get_bn_mask(index, filter_index)
    
    
    def set_pruning_layer_and_ratio(self, args, prune_rate) -> None:
        """ this function determines the index of target layers and pruning ratio of each layer """
        
        if 'vgg' in args.arch:
            
            for index, (name, module) in enumerate(self.model.named_modules()):
                
                if isinstance(module, nn.Conv2d) or isinstance(module, nn.BatchNorm2d):
                    
                    self.prune_rate[index] = prune_rate
                    self.mask_index.append(index)
                                
        elif "resnet" in args.arch:
            
            for index, (name, module) in enumerate(self.model.named_modules()):

                if isinstance(module, nn.Conv2d) or isinstance(module, nn.BatchNorm2d):
                    
                    self.mask_index.append(index)
                    self.prune_rate[index] = prune_rate
                
            # ---
            if args.skip_downsample:      # skip downsample layer
            
                skip_list = []
                for idx, (name, module) in enumerate(self.model.named_modules()):
                    if ('downsample') in name and (isinstance(module, nn.Conv2d) or isinstance(module, nn.BatchNorm2d)):
                        skip_list.append(idx)
            
                for x in skip_list:
                    
                    self.prune_rate.pop(x)
                    self.mask_index.remove(x)
                    
        else:     # --- the channel selection is independent for shortcut-pathway and residual-pathway
            
            raise ValueError
    

    
class FPGM_Mask(SFP_Mask_Base):
    """
        this class only process with distance, or similarity, based components
        
        this function provides a very rough method to obtain kernels that 'might' be close to the geometric median
    """
    
    def __init__(self, args, model) -> None:
       
        super().__init__(model)
       
        self.prune_rate_distance = {}
        self.mask_dict_distance = {}
        self.mask_index_distance = []
    
    
    def mask_initialization(self, args, **kwargs) -> None:
        
        self.set_pruning_layer_and_ratio(args)
        
        self.mask_generation(args)

    
    def mask_generation(self, args, remove_bn=True, allow_intersection=False) -> None:
        """
            remove_bn: when True, remove following BN for every Conv
            allow_intersection: when True, calculate norm_based_index and distance_based_index independetly
        """

        for index, (name, module) in enumerate(self.model.named_modules()):     # --- for each layer
        
            # --- for mask based on norm
            if index in self.mask_index:
                
                if isinstance(module, nn.Conv2d):
                    
                    self.mask_dict[index], filter_index = self.get_conv_mask_by_norm(module.weight.data, index, return_idx=True)
            
                if remove_bn and isinstance(module, nn.BatchNorm2d) and (filter_index is not None):

                    self.mask_dict[index] = self.get_bn_mask(index, filter_index)
                
            # --- for mask based on distance
            if index in self.mask_index_distance:
                
                target_tensor = module.weight.data

                if not allow_intersection and (filter_index is not None):
                    
                    dummy_index = torch.arange(target_tensor.shape[0], device=target_tensor.device)
                    keep_indices = dummy_index[~torch.isin(dummy_index, filter_index)]
                    target_tensor = torch.index_select(target_tensor, 0, keep_indices)
            
                if isinstance(module, nn.Conv2d):

                    filter_index_distance_reduced = (
                                                    self.get_conv_mask_by_distance(target_tensor, index)
                                                    if not allow_intersection and (filter_index is not None)
                                                    else self.get_conv_mask_by_distance(module.weight.data, index)
                                                    )
                    
                    filter_index_distance = (
                                            keep_indices[filter_index_distance_reduced]
                                            if not allow_intersection and (filter_index is not None)
                                            else filter_index_distance_reduced
                                            )
                    
                    self.mask_dict_distance[index] = self.get_mask(module.weight.data, filter_index_distance)
            
                if remove_bn and isinstance(module, nn.BatchNorm2d) and (filter_index_distance is not None):
                    
                    self.mask_dict_distance[index] = self.get_bn_mask(index, filter_index_distance)

    
    def get_conv_mask_by_distance(self, input, index, return_idx=False) -> torch.Tensor:
        """ the input size has been reduced """
        
        return self.get_filter_indices_by_distance(input, index)
        
        
    def get_filter_indices_by_distance(self, input, index) -> torch.Tensor:
        """
            input: tensor, weight of kernel
            index: layer index
            
            current design only uses euclidean distance, in the original code, they also adopted cosine distance
            intuitively, a lot of 'similarity' metrics can be used for such task
        """
        
        assert input.ndim == 4
        
        similar_pruned_num = int(input.shape[0] * self.prune_rate_distance[index])
        input = input.view(input.shape[0], -1)

        similar_matrix = torch.cdist(input, input, p=2)
        similar_sum = torch.sum(torch.abs(similar_matrix), dim=0)

        return torch.argsort(similar_sum)[:similar_pruned_num]
    
    
    def set_pruning_layer_and_ratio(self, args) -> None:
        
        if 'vgg' in args.arch:
            
            for index, (name, module) in enumerate(self.model.named_modules()):
                
                if isinstance(module, nn.Conv2d) or isinstance(module, nn.BatchNorm2d):
                    
                    self.prune_rate[index] = args.prune_rate
                    self.prune_rate_distance[index] = args.prune_rate_distance
                    self.mask_index.append(index)
                    self.mask_index_distance.append(index)
  
        elif "resnet" in args.arch:
            
            for index, (name, module) in enumerate(self.model.named_modules()):

                if isinstance(module, nn.Conv2d) or isinstance(module, nn.BatchNorm2d):
                    
                    self.mask_index.append(index)
                    self.mask_index_distance.append(index)
                    self.prune_rate[index] = args.prune_rate
                    self.prune_rate_distance[index] = args.prune_rate_distance
                
            # ---
            if args.skip_downsample:
            
                skip_list = []
                for idx, (name, module) in enumerate(self.model.named_modules()):
                    if ('downsample') in name and (isinstance(module, nn.Conv2d) or isinstance(module, nn.BatchNorm2d)):
                        skip_list.append(idx)
            
                for x in skip_list:
                    
                    self.prune_rate.pop(x)
                    self.prune_rate_distance.pop(x)
                    self.mask_index.remove(x)
                    self.mask_index_distance.remove(x)
                    
        else:
            
            raise ValueError
    
    
    def mask_apply(self, remove_grad=True) -> None:
        """ 
            apply the mask to the weight value based on mask_distance
        """
        
        def _apply(_mask):
            
            module.weight.data = module.weight.data * _mask
            
            if hasattr(module, 'bias') and (getattr(module, 'bias') is not None):
                
                module.bias.data = module.bias.data * _mask
            
            if remove_grad and (module.weight.grad is not None):
                
                module.weight.grad.data = module.weight.grad.data * _mask
                
                if hasattr(module, 'bias') and (getattr(module, 'bias') is not None):
                    
                    module.bias.grad.data = module.bias.grad.data * _mask
        
        for index, (name, module) in enumerate(self.model.named_modules()):     # --- for each layer
            
            if index in self.mask_index:
                
                _apply(self.mask_dict[index])
            
            if index in self.mask_index_distance:
                
                _apply(self.mask_dict_distance[index])

    

class AFPGM_Mask(FPGM_Mask):
    """ ASFP + FPGM """
    
    def __init__(self, args, model) -> None:
       
        super().__init__(args, model)
        
        self.dynamic_pruning_rate = self.dynamic_pruning_rate_calculation(args.epochs, args.prune_rate)
        self.dynamic_pruning_rate_distance = self.dynamic_pruning_rate_calculation(args.epochs, args.prune_rate_distance)
        
    
    def mask_initialization(self, args, epoch=0, **kwargs) -> None:
       
        self.set_pruning_layer_and_ratio(args, self.dynamic_pruning_rate[epoch], self.dynamic_pruning_rate_distance[epoch])
        self.mask_generation(args)
        
        
    def set_pruning_layer_and_ratio(self, args, prune_rate, prune_rate_distance) -> None:
        
        if 'vgg' in args.arch:
            
            for index, (name, module) in enumerate(self.model.named_modules()):
                
                if isinstance(module, nn.Conv2d) or isinstance(module, nn.BatchNorm2d):
                    
                    self.prune_rate[index] = prune_rate
                    self.prune_rate_distance[index] = prune_rate_distance
                    self.mask_index.append(index)
                    self.mask_index_distance.append(index)
  
        elif "resnet" in args.arch:
            
            for index, (name, module) in enumerate(self.model.named_modules()):

                if isinstance(module, nn.Conv2d) or isinstance(module, nn.BatchNorm2d):
                    
                    self.mask_index.append(index)
                    self.mask_index_distance.append(index)
                    self.prune_rate[index] = prune_rate
                    self.prune_rate_distance[index] = prune_rate_distance
                
            # ---
            if args.skip_downsample:
            
                skip_list = []
                for idx, (name, module) in enumerate(self.model.named_modules()):
                    if ('downsample') in name and (isinstance(module, nn.Conv2d) or isinstance(module, nn.BatchNorm2d)):
                        skip_list.append(idx)
            
                for x in skip_list:
                    
                    self.prune_rate.pop(x)
                    self.prune_rate_distance.pop(x)
                    self.mask_index.remove(x)
                    self.mask_index_distance.remove(x)
                    
        else:
            
            raise ValueError