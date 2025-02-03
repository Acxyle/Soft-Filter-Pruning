#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TODO:
    
    remove the 0-value kernels by generate new nn.Module, method inherits from CP_Pruning

"""


from typing import Literal, Any

import torch
import torch.nn as nn

from spikingjelly.activation_based import functional, layer


# ----------------------------------------------------------------------------------------------------------------------
def get_channel_index(kernel:torch.Tensor, p=1):
    """
        calculate the indces of the non-zerovalue channels
    
        input:
            kernel: 
            num_elimination: 
            
        parameter:
            p: the order of the norm (default, p=1)
    """
    
    sum_of_kernel = torch.norm(kernel.view(kernel.shape[0], -1), p=p, dim=1)
    
    return torch.where(sum_of_kernel!=0)[0]


def get_new_conv(conv:nn.Module, dim:Literal[0,1], channel_index:torch.Tensor, frame='torch', step_mode='m', device='cuda:0') -> Any:
    """
        This function returns a new Conv2d layer with reduced channels based on channel_index.
        no grad remaining
    """
    
    if frame == 'torch':
        framework = nn
    elif frame == 'spikingjelly':
        framework = layer
    
    has_bias = conv.bias is not None
    
    if dim == 0:     # --- (out, in, k1, k2) -> (out', in, k1, k2)
        new_out_channels = channel_index.numel()
        new_in_channels = conv.in_channels
    elif dim == 1:     # --- (out, in, k1, k2) -> (out, in', k1, k2)
        new_out_channels = conv.out_channels
        new_in_channels = channel_index.numel()
    else:
        raise ValueError("dim must be 0 (out_channels) or 1 (in_channels)")
    
    # --- create empty Conv Module
    new_conv = framework.Conv2d(in_channels=new_in_channels,
                         out_channels=new_out_channels,
                         kernel_size=conv.kernel_size,
                         stride=conv.stride,
                         padding=conv.padding,
                         dilation=conv.dilation,
                         bias=has_bias)
    
    # --- load non-zero values to new Conv Module
    new_conv.weight.data, _ = split_kernel_along_dim(conv.weight.data, dim, channel_index, device=device)
    new_conv.weight.requires_grad = conv.weight.requires_grad
    
    if has_bias and dim == 0:
        new_conv.bias.data, _ = split_kernel_along_dim(conv.bias.data, dim, channel_index, device=device)
        new_conv.bias.requires_grad = conv.bias.requires_grad
    elif has_bias:
        new_conv.bias.data = conv.bias.data
        new_conv.bias.requires_grad = conv.bias.requires_grad
    
    if frame == 'spikingjelly':
        functional.set_step_mode(new_conv, step_mode=step_mode)
    
    return new_conv


def get_new_linear(linear:nn.Module, dim:Literal[0,1], channel_index:torch.Tensor, avgp=None, frame='torch', step_mode='m', device='cuda:0') -> Any:
    """
        channel_index: maintained channels
    """
    
    if frame == 'torch':
        framework = nn
    elif frame == 'spikingjelly':
        framework = layer
    
    # --- kernel -> feature
    if dim == 0:
        
        new_linear = framework.Linear(in_features = linear.in_features,
                                    out_features = channel_index.numel(),
                                    bias = linear.bias is not None)
        
        new_linear.weight.data, _ = split_kernel_along_dim(linear.weight.data, dim, channel_index, device=device)
        new_linear.bias.data, _ = split_kernel_along_dim(linear.bias.data, dim, channel_index, device=device)
        
    # --- feature - > kernel
    elif dim == 1:
        
        if avgp is not None:
        
            dummy_tensor = torch.empty(512, 7, 7, dtype=torch.long, device=device)     # --- assume the channels of last conv is 512
            
            for c in range(dummy_tensor.shape[0]):
                
                if c in channel_index:
                    dummy_tensor[c, :, :] = 1
                else:
                    dummy_tensor[c, :, :] = -1
            
            dummy_tensor = dummy_tensor.reshape(-1)
            
            channel_index = torch.where(dummy_tensor==1)[0]
        
        # ---
        new_linear = framework.Linear(in_features = channel_index.numel(),
                                    out_features = linear.out_features,
                                    bias = linear.bias is not None)
        
        new_linear.weight.data, _ = split_kernel_along_dim(linear.weight.data, 1, channel_index, device=device)
        new_linear.bias.data = linear.bias.data
        
    else:
        
        raise ValueError
    
    if frame == 'spikingjelly':
        functional.set_step_mode(new_linear, step_mode=step_mode)
    
    return new_linear


def get_new_norm(norm:nn.Module, channel_index:torch.Tensor, frame='torch', step_mode='m', device='cuda:0') -> Any:
    """
        input:
            norm: nn.BatchNorm
            channel_index
    """
    
    if frame == 'torch':
        framework = nn
    elif frame == 'spikingjelly':
        framework = layer
    
    assert isinstance(norm, (framework.BatchNorm1d, framework.BatchNorm2d, framework.BatchNorm3d))
    assert torch.all(torch.where(norm.weight.data!=0)[0] == channel_index)

    num_features = channel_index.numel()
    
    new_norm = type(norm)(num_features=num_features,
                              eps=norm.eps,
                              momentum=norm.momentum,
                              affine=norm.affine,
                              track_running_stats=norm.track_running_stats)

    def _update_param(attr):
        param, _ = split_kernel_along_dim(getattr(norm, attr).data, 0, channel_index, device=device)
        setattr(new_norm, attr, nn.Parameter(param))

    _update_param('weight')
    _update_param('bias')

    if norm.track_running_stats:
        new_norm.running_mean, _ = split_kernel_along_dim(norm.running_mean.data, 0, channel_index, device=device)
        new_norm.running_var, _ = split_kernel_along_dim(norm.running_var.data, 0, channel_index, device=device)
    
    if frame == 'spikingjelly':
        functional.set_step_mode(new_norm, step_mode=step_mode)
        
    return new_norm


def split_kernel_along_dim(input:torch.Tensor, dim:Literal[0,1], channel_index:torch.Tensor, device='cuda:0') -> (torch.Tensor):
    """
        input:
            input: weight or bias value of kernel
            dim: 0 for out_channel and 1 for in_channel
            channel_index: the indices of the unwanted channel
            
    """

    all_indices = torch.arange(input.shape[dim], device=device)
    
    selected_tensor = torch.index_select(input, dim, channel_index)
    
    removed_mask = ~torch.isin(all_indices, channel_index)
    removed_tensor = torch.index_select(input, dim, all_indices[removed_mask])
    
    return selected_tensor, removed_tensor


