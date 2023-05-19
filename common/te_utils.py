
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from contextlib import contextmanager, nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F

import transformer_engine.pytorch as te
from transformer_engine.common.recipe import Format, DelayedScaling


class FP8Linear(nn.Module):
    """This is a wrapper of te.Linear, which pads the input to a multiple of 16 and then truncates the output to the original size."""
    TE_BASE = 16
    
    def __init__(self, in_features, out_features, bias=True, params_dtype=torch.float32):
        """Constructor of FP8Linear.

        Args:
            in_features (int): size of each input sample.
            out_features (int): size of each output sample.
            bias (bool): If set to ``False``, the layer will not learn an additive bias. Default: ``True``.
            params_dtype (torch.dtype): The data type of the weight and bias. Default: ``torch.float32``.
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        pad_in_features = FP8Linear.Round2Times(in_features, FP8Linear.TE_BASE)
        pad_out_features = FP8Linear.Round2Times(out_features, FP8Linear.TE_BASE)
        self.in_padding = pad_in_features - in_features
        self.linear = te.Linear(pad_in_features, pad_out_features, bias=bias, params_dtype=params_dtype)

    def forward(self, x):
        """Forward pass of FP8Linear.

        Args:
            x (torch.Tensor): Input tensor of shape ``[batch_size, ..., in_features]``.

        Returns:
            torch.Tensor: Output tensor of shape ``[batch_size, ..., out_features]``.
        """
        return self.padding_block_io(self.linear, x)

    @staticmethod
    def Round2Times(value, base):
        """Round up the value to the nearest multiple of base.

        Args:
            value (int): The value to be rounded.

        Returns:
            int: The rounded value.
        """
        return (value + base - 1) // base * base

    def padding_block_io(self, block, x):
        """Pad the input to a multiple of 16 and then truncate the output to the original size.

        Args:
            block (torch.nn.Module): The block to be padded.
            x (torch.Tensor): Input tensor of shape ``[batch_size, ..., in_features]``.

        Returns:
            torch.Tensor: Output tensor of shape ``[batch_size, ..., out_features]``.
        """
        shape = list(x.shape)
        last_dim = shape[-1]
        # reshape to 2 dims
        x = x.view(-1, last_dim)
        first_dim = len(x)
        pad_first_dim = FP8Linear.Round2Times(first_dim, FP8Linear.TE_BASE)
        first_dim_padding = pad_first_dim - first_dim
        if first_dim_padding > 0 or self.in_padding > 0:
            x = F.pad(x, (0, self.in_padding, 0, first_dim_padding))
        x = block(x)
        x = x[:first_dim, :self.out_features]
        shape[-1] = self.out_features
        return x.reshape(shape)

    @property
    def weight(self):
        return self.linear.weight[:self.out_features, :self.in_features]

    @weight.setter
    def weight(self, value):
        raise NotImplementedError

    @property
    def bias(self):
        if self.linear.bias is None:
            return None
        return self.linear.bias[:self.out_features]

    @bias.setter
    def bias(self, value):
        raise NotImplementedError


@torch.no_grad()
def replace_with_telinear(model):
    """
    Replace torch.nn.Linear with FP8Linear in a model.

    Args:
        model (torch.nn.Module): The model to be replaced.

    Returns:
        torch.nn.Module: The model with FP8Linear.
    """
    model = model.cuda()
    def _replace_with_telinear(model):
        if isinstance(model, torch.nn.Linear):
            if getattr(model, 'use_fp32_linear', False):
                return model
            te_linear = build_telinear_from_linear(model)
            return te_linear
        else:
            for name, module in model.named_children():
                setattr(model, name, _replace_with_telinear(module))
            return model
    model = _replace_with_telinear(model)
    return model


@torch.no_grad()
def build_telinear_from_linear(linear):
    """build FP8Linear from torch.nn.Linear.

    Args:
        linear (torch.nn.Linear): The torch.nn.Linear to be replaced.

    Returns:
        FP8Linear: The FP8Linear with same input and output features.
    """
    weight_dtype = torch.float32
    te_linear = FP8Linear(
            in_features=linear.in_features,
            out_features=linear.out_features,
            bias=linear.bias is not None,
            params_dtype=weight_dtype,
    ).cuda()
    te_linear.weight[:linear.out_features, :linear.in_features].copy_(linear.weight.to(te_linear.weight.dtype))
    if linear.bias is not None:
        te_linear.bias[:linear.out_features].copy_(linear.bias.to(te_linear.bias.dtype))
    return te_linear


class TeUtils:
    """A utility class for using transformer_engine."""
    @staticmethod
    def get_fp8_recipe(format, max_history_len, amax_compute_algo):
        """Get the recipe for FP8.

        Args:
            format (str): The format of FP8. It can be 'hybrid', 'e4m3' or 'e5m2'.
            max_history_len (int): The maximum length of history.
            amax_compute_algo (str): The algorithm to compute amax. It can be 'max' or 'most_recent'.

        Returns:
            DelayedScaling: The recipe for FP8.
        """
        assert format in ['hybrid', 'e4m3', 'e5m2']
        if format == 'hybrid':
            fp8_format = Format.HYBRID  # E4M3 during forward pass, E5M2 during backward pass
        elif format == 'e4m3':
            fp8_format = Format.E4M3
        elif format == 'e5m2':
            fp8_format = Format.E5M2

        fp8_recipe = DelayedScaling(
            fp8_format=fp8_format,
            amax_history_len=max_history_len,
            amax_compute_algo=amax_compute_algo)
        return fp8_recipe

    @staticmethod
    def get_autocast(amp_enable, fp8_enable, fp8_format='hybrid', max_history_len=16, amax_compute_algo='max'):
        """Get the context manager for autocast.
        
        Args:
            amp_enable (bool): If set to ``True``, the amp autocast will be enabled.
            fp8_enable (bool): If set to ``True``, the fp8 autocast will be enabled.
            fp8_format (str): The format of FP8. It can be 'hybrid', 'e4m3' or 'e5m2'.
            max_history_len (int): The maximum length of history.
            amax_compute_algo (str): The algorithm to compute amax. It can be 'max' or 'most_recent'.

        Returns:
            contextmanager: The context manager for autocast.
        """
        autocast_context = lambda : torch.cuda.amp.autocast(enabled=amp_enable, dtype=torch.bfloat16)
        if fp8_enable:
            fp8_recipe = TeUtils.get_fp8_recipe(fp8_format, max_history_len, amax_compute_algo)
            fp8_context = lambda : te.fp8_autocast(enabled=fp8_enable, fp8_recipe=fp8_recipe)
        else:
            fp8_context = lambda : nullcontext()

        @contextmanager
        def context_manager(*args, **kwargs):
            with autocast_context():
                with fp8_context():
                    yield
            
        return context_manager