"""
    Mamba Module 
    Reference: https://github.com/SarthakYadav/audio-mamba-official/blob/master/src/models/mamba_ssast.py
"""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional
from functools import partial
from s3prl.upstream.wav2vec2.wav2vec2_model import (
    ConvFeatureExtractionModel,
    GradMultiply,
    SamePad,
)
from mamba_ssm.modules.mamba_simple import Mamba

def get_activation_fn(activation: str):
    """Returns the activation function corresponding to `activation`"""

    def gelu_accurate(x):
        if not hasattr(gelu_accurate, "_a"):
            gelu_accurate._a = math.sqrt(2 / math.pi)
        return (
            0.5
            * x
            * (1 + torch.tanh(gelu_accurate._a * (x + 0.044715 * torch.pow(x, 3))))
        )

    def gelu(x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.gelu(x.float()).type_as(x)

    if activation == "relu":
        return F.relu
    elif activation == "relu_squared":
        return relu_squared
    elif activation == "gelu":
        return gelu
    elif activation == "gelu_fast":
        return gelu_accurate
    elif activation == "gelu_accurate":
        return gelu_accurate
    elif activation == "tanh":
        return torch.tanh
    elif activation == "linear":
        return lambda x: x
    elif activation == "swish":
        return torch.nn.SiLU
    else:
        raise RuntimeError("--activation-fn {} not supported".format(activation))

class SplitLinear(nn.Module):
    """Split Linear Layer"""

    def __init__(self, in_dim, in_split, out_dim):
        super().__init__()

        self.in_dim = in_dim  # Din
        self.in_split = in_split  # N
        self.out_dim = out_dim  # Dout

        if in_split > 1:
            # weight = torch.zeros((1, 1, self.in_split, self.in_dim, self.out_dim))
            weight = torch.zeros((self.in_split, self.in_dim, self.out_dim))
            self.weight = nn.Parameter(weight, requires_grad=True)
            nn.init.uniform_(self.weight, -(self.in_dim**-0.5), self.in_dim**-0.5)

            bias = torch.zeros((1, 1, self.in_split, self.out_dim))
            self.bias = nn.Parameter(bias, requires_grad=True)
            nn.init.uniform_(self.bias, -(self.in_dim**-0.5), self.in_dim**-0.5)
        else:
            self.layer = nn.Linear(self.in_dim, self.out_dim)

    def forward(self, x: torch.Tensor):
        # x: shape = B x T x NDin

        if self.in_split == 1:
            return self.layer(x)
        else:
            x = x.reshape(x.shape[0], x.shape[1], self.in_split, 1, self.in_dim)
            # x: B x T x N x 1 x Din

            out = torch.einsum("...klm,kmn->...kln", x, self.weight).squeeze(3)
            # out: B x T x N x Dout
            out = out + self.bias

            return out.reshape(x.shape[0], x.shape[1], -1)

class MambaBlockAudioMamba(nn.Module):
    def __init__(
        self, 
        dim, 
        mixer_cls, 
        norm_cls=nn.LayerNorm, 
        residual_in_fp32=False, 
        ffn_dim=0,
        activation_fn = "relu",
        **kwargs,
    ):
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.mixer = mixer_cls(dim)
        self.norm = norm_cls(dim)
        self.ffn_dim = ffn_dim
        if self.ffn_dim > 0:
            self.ffn_layer_norm = norm_cls(dim)
            self.activation_fn = get_activation_fn(activation_fn)
            self.fc1 = nn.Linear(dim, ffn_dim)
            self.fc2 = nn.Linear(ffn_dim, dim)

    def forward(
        self, hidden_states: Tensor, residual: Optional[Tensor] = None, inference_params=None
    ):
        if residual is None:
            residual = hidden_states
        else:
            residual = residual + hidden_states
            
        hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
        
        if self.residual_in_fp32:
            residual = residual.to(torch.float32)

        hidden_states = self.mixer(hidden_states, inference_params=inference_params)

        if self.ffn_dim > 0:
            residual = residual + hidden_states
            x = self.ffn_layer_norm(residual)
            x = self.activation_fn(self.fc1(x))
            x = self.fc2(x)

        return hidden_states, residual

class BiMambaBlockAudioMamba(nn.Module):
    def __init__(
        self, 
        dim, 
        mixer_cls, 
        norm_cls=nn.LayerNorm, 
        residual_in_fp32=False, 
        ffn_dim=0,
        activation_fn = "relu",
        **kwargs,
    ):
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.mixer = mixer_cls(dim)
        self.mixer_rev = mixer_cls(dim)
        self.norm = norm_cls(dim)
        self.ffn_dim = ffn_dim
        if self.ffn_dim > 0:
            self.ffn_layer_norm = norm_cls(dim)
            self.activation_fn = get_activation_fn(activation_fn)
            self.fc1 = nn.Linear(dim, ffn_dim)
            self.fc2 = nn.Linear(ffn_dim, dim)

    def forward(
        self, hidden_states: Tensor, residual: Optional[Tensor] = None, inference_params=None
    ):
        if residual is None:
            residual = hidden_states
        else:
            residual = residual + hidden_states
            
        hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
        
        if self.residual_in_fp32:
            residual = residual.to(torch.float32)

        hidden_states_rev = torch.flip(hidden_states, dims=(1,))
        hidden_states = self.mixer(hidden_states, inference_params=inference_params)
        hidden_states_rev = self.mixer_rev(hidden_states_rev, inference_params=inference_params)
        hidden_states = hidden_states + torch.flip(hidden_states_rev, dims=(1, ))

        if self.ffn_dim > 0:
            residual = residual + hidden_states
            x = self.ffn_layer_norm(residual)
            x = self.activation_fn(self.fc1(x))
            x = self.fc2(x)

        return hidden_states, residual

class MambaBlock(nn.Module):
    def __init__(
        self, 
        dim, 
        mixer_cls, 
        norm_cls=nn.LayerNorm, 
        residual_in_fp32=False, 
        ffn_dim=0,
        activation_fn = "relu",
        dropout=0.1,
        activation_dropout=0.1,
    ):
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.mixer = mixer_cls(dim)
        self.dropout1 = nn.Dropout(dropout)
        self.norm = norm_cls(dim)
        self.ffn_dim = ffn_dim
        if self.ffn_dim > 0:
            self.activation_fn = get_activation_fn(activation_fn)
            self.fc1 = nn.Linear(dim, ffn_dim)
            self.fc2 = nn.Linear(ffn_dim, dim)
            self.dropout2 = nn.Dropout(activation_dropout)
            self.dropout3 = nn.Dropout(dropout)
            self.final_layer_norm = norm_cls(dim)

    def forward(
        self, x: Tensor, inference_params=None
    ):
        r"""Pass the input through the encoder layer.

        Args:
            hidden_states: the sequence to the encoder layer (required).
            residual: hidden_states = Mixer(LN(residual))
        """
        residual = x
        x = self.mixer(x, inference_params=inference_params)
        layer_result = x
        x = self.dropout1(x)
        x = residual + x
        x = self.norm(x.to(dtype=self.norm.weight.dtype))
        if self.ffn_dim > 0:
            residual = x
            x = self.activation_fn(self.fc1(x))
            x = self.dropout2(x)
            x = self.fc2(x)
            layer_result = x
            x = self.dropout3(x)
            x = residual + x
            x = self.final_layer_norm(x)
        return x, layer_result

class BiMambaBlock(nn.Module):
    def __init__(
        self, 
        dim, 
        mixer_cls, 
        norm_cls=nn.LayerNorm, 
        residual_in_fp32=False, 
        ffn_dim=0,
        activation_fn = "relu",
        dropout = 0.1,
        activation_dropout = 0.1,
    ):
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.mixer = mixer_cls(dim)
        self.mixer_rev = mixer_cls(dim)
        self.dropout1 = nn.Dropout(dropout)
        self.norm = norm_cls(dim)
        self.ffn_dim = ffn_dim
        if self.ffn_dim > 0:
            self.activation_fn = get_activation_fn(activation_fn)
            self.fc1 = nn.Linear(dim, ffn_dim)
            self.fc2 = nn.Linear(ffn_dim, dim)
            self.dropout2 = nn.Dropout(activation_dropout)
            self.dropout3 = nn.Dropout(dropout)
            self.final_layer_norm = norm_cls(dim)

    def forward(
        self, x: Tensor, inference_params=None
    ):
        r"""Pass the input through the encoder layer.

        Args:
            hidden_states: the sequence to the encoder layer (required).
            residual: hidden_states = Mixer(LN(residual))
        """
        residual = x
        x_rev = torch.flip(x, dims=(1,))
        x = self.mixer(x, inference_params=inference_params)
        x_rev = self.mixer_rev(x_rev)
        x = x + torch.flip(x_rev, dims=(1, ))
        layer_result = x
        x = self.dropout1(x)
        x = residual + x
        x = self.norm(x.to(dtype=self.norm.weight.dtype))
        if self.ffn_dim > 0:
            residual = x
            x = self.activation_fn(self.fc1(x))
            x = self.dropout2(x)
            x = self.fc2(x)
            layer_result = x
            x = self.dropout3(x)
            x = residual + x
            x = self.final_layer_norm(x)

        return x, layer_result

def create_block(
    d_model,
    direction='uni',
    mamba_type='mamba',
    ffn_dim=0,
    activation_fn="relu",
    dropout=0.1,
    activation_dropout=0.1,
    ssm_cfg=None,
    norm_epsilon=1e-6,
    residual_in_fp32=False,
    layer_idx=None,
    device=None,
    dtype=None
):
    if ssm_cfg is None:
        ssm_cfg = {}
    factory_kwargs = {"device": device, "dtype": dtype}
    if mamba_type == 'mamba':
        mixer_cls = partial(Mamba, layer_idx=layer_idx, **ssm_cfg, **factory_kwargs)
    elif mamba_type == 'mamba2':
        # mixer_cls = partial(Mamba2Simple, layer_idx=layer_idx, **ssm_cfg, **factory_kwargs)
        raise NotImplementedError
    else:
        raise NotImplementedError
    norm_cls = partial(
        nn.LayerNorm, eps=norm_epsilon, **factory_kwargs
    )
    if direction == 'uni':
        block = MambaBlock(
            d_model,
            mixer_cls,
            norm_cls=norm_cls,
            residual_in_fp32=residual_in_fp32,
            ffn_dim=ffn_dim,
            activation_fn=activation_fn,
            dropout=dropout,
            activation_dropout=activation_dropout,
        )
    elif direction == 'bi':
        block = BiMambaBlock(
            d_model,
            mixer_cls,
            norm_cls=norm_cls,
            residual_in_fp32=residual_in_fp32,
            ffn_dim=ffn_dim,
            activation_fn=activation_fn,
            dropout=dropout,
            activation_dropout=activation_dropout,
        )
    elif direction == 'uni-audio-mamba':
        block = MambaBlockAudioMamba(
            d_model,
            mixer_cls,
            norm_cls=norm_cls,
            residual_in_fp32=residual_in_fp32,
            ffn_dim=ffn_dim,
            activation_fn=activation_fn,
            dropout=dropout,
            activation_dropout=activation_dropout,
        )
    elif direction == 'bi-audio-mamba':
        block = BiMambaBlockAudioMamba(
            d_model,
            mixer_cls,
            norm_cls=norm_cls,
            residual_in_fp32=residual_in_fp32,
            ffn_dim=ffn_dim,
            activation_fn=activation_fn,
            dropout=dropout,
            activation_dropout=activation_dropout,
        )
    else:
        raise NotImplementedError
    block.layer_idx = layer_idx
    return block

# https://github.com/huggingface/transformers/blob/c28d04e9e252a1a099944e325685f14d242ecdcd/src/transformers/models/gpt2/modeling_gpt2.py#L454
def _init_weights(
    module,
    n_layer,
    initializer_range=0.02,  # Now only used for embedding layer.
    rescale_prenorm_residual=True,
    n_residuals_per_layer=1,  # Change to 2 if we have MLP
):
    if isinstance(module, nn.Linear):
        if module.bias is not None:
            if not getattr(module.bias, "_no_reinit", False):
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
        #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
        #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
        #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
        #
        # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                # Following Pytorch init, except scale by 1/sqrt(2 * n_layer)
                # We need to reinit p since this code could be called multiple times
                # Having just p *= scale would repeatedly scale it down
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)

class MambaEncoder(nn.Module):
    def __init__(
        self,
        embed_dim=768,
        direction='uni',
        mamba_type='mamba',
        ffn_dim=0,
        activation_fn="relu",
        dropout=0.1,
        activation_dropout=0.1,
        depth=12,
        ssm_cfg=None,
        residual_in_fp32: bool = False,
        conv_pos=128,
        conv_pos_groups=16,
        norm_epsilon: float = 1e-6,
        device=None,
        dtype=None,
    ):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.factory_kwargs = factory_kwargs
        self.residual_in_fp32 = residual_in_fp32
        self.depth = depth
        # Positional encoding same as HuBERT
        self.pos_conv = nn.Conv1d(
            embed_dim,
            embed_dim,
            kernel_size=conv_pos,
            padding=conv_pos // 2,
            groups=conv_pos_groups,
        )
        # Initialize the weights of positional encoding
        dropout = 0
        std = math.sqrt((4 * (1.0 - dropout)) / (conv_pos * embed_dim))
        nn.init.normal_(self.pos_conv.weight, mean=0, std=std)
        nn.init.constant_(self.pos_conv.bias, 0)
        self.pos_conv = nn.utils.weight_norm(self.pos_conv, name="weight", dim=2)
        self.pos_conv = nn.Sequential(self.pos_conv, SamePad(conv_pos), nn.GELU())
        self.direction = direction
        # Mamba encoder blocks
        self.blocks = nn.ModuleList()
        for i in range(depth):
            self.blocks.append(
                create_block(
                    embed_dim,
                    direction=direction,
                    mamba_type=mamba_type,
                    ffn_dim=ffn_dim,
                    activation_fn=activation_fn,
                    dropout=dropout,
                    activation_dropout=activation_dropout,
                    ssm_cfg=ssm_cfg,
                    norm_epsilon=norm_epsilon,
                    residual_in_fp32=residual_in_fp32,
                    layer_idx=i,
                    **self.factory_kwargs,
                )
            )
        self.last_norm = partial(nn.LayerNorm, eps=norm_epsilon)(embed_dim)

        self.initialize_weights()

    def initialize_weights(self):
        # pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], self.grid_size(), cls_token=self.use_cls_token)
        # self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # if self.use_cls_token:
        #     torch.nn.init.normal_(self.cls_token, std=0.02)
        # torch.nn.init.normal_(self.mask_token, std=0.02)
        self.apply(
            partial(
                _init_weights,
                n_layer=self.depth,
            )
        )
    
    def forward(self, x, padding_mask=None, get_hidden=False, inference_params=None):
        if padding_mask is not None:
            x[padding_mask] = 0
        # Forward positional encoding
        x_conv = self.pos_conv(x.transpose(1, 2))
        x_conv = x_conv.transpose(1, 2)
        x = x + x_conv

        residual = None
        hidden_states = x
        
        layer_hiddens = []
        for layer in self.blocks:
            if self.direction == 'uni-audio-mamba':
                x, residual = layer(
                    x, residual, inference_params=inference_params
                )
            else:
                x, layer_result = layer(
                    x, inference_params=inference_params
                )
            if get_hidden:
                layer_hiddens.append(x)

        # Add residual for the last layer hidden states
        if residual is None:
            residual = hidden_states
        else:
            residual = residual + hidden_states
        hidden_states = self.last_norm(residual.to(dtype=self.last_norm.weight.dtype))

        x = hidden_states

        return x, layer_hiddens 