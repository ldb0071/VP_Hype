#!/usr/bin/env python3

# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


import torch
import torch.nn as nn
from timm.models.registry import register_model
import math
from timm.models.layers import trunc_normal_, DropPath, LayerNorm2d
from timm.models._builder import resolve_pretrained_cfg
try:
    from timm.models._builder import _update_default_kwargs as update_args
except:
    from timm.models._builder import _update_default_model_kwargs as update_args
from timm.models.vision_transformer import Mlp, PatchEmbed
from timm.models.layers import DropPath, trunc_normal_
from timm.models.registry import register_model
import torch.nn.functional as F
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
from einops import rearrange, repeat
from .registry import register_pip_model
# Import prompt modules from MP_HSIR
from .MP_HSIR import Text_Prompt, TVSP, PromptFusion
# Prefer local EfficientKAN implementation; fallback to external ones if needed
try:
    from .efficient_kan import EfficientKAN as LocalEfficientKAN
    _KAN_BACKEND = "local_efficient_kan"
except Exception:
    LocalEfficientKAN = None
    try:
        from efficientkan import KANLinear  # pip install efficientkan
        _KAN_BACKEND = "efficientkan"
    except Exception:
        try:
            from efficient_kan import EfficientKAN  # pip install git+https://github.com/Blealtan/efficient-kan.git
            _KAN_BACKEND = "efficient_kan"
        except Exception:
            _KAN_BACKEND = None
from pathlib import Path

# Define a reusable KAN-based MLP only when KANLinear backend is available
if '_KAN_BACKEND' in globals() and _KAN_BACKEND == "efficientkan":
    class KAN_MLP(nn.Module):
        """KAN-based MLP wrapper using KANLinear (two linear KAN layers with optional dropout)."""
        def __init__(self, in_features, hidden_features, out_features, grid_size=5, spline_order=3, dropout=0.):
            super().__init__()
            self.fc1 = KANLinear(in_features, hidden_features, grid_size=grid_size, spline_order=spline_order)
            self.drop1 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
            self.fc2 = KANLinear(hidden_features, out_features, grid_size=grid_size, spline_order=spline_order)
            self.drop2 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        def forward(self, x):
            x = self.fc1(x)
            x = self.drop1(x)
            x = self.fc2(x)
            x = self.drop2(x)
            return x
else:
    KAN_MLP = None


def window_partition(x, window_size):
    """
    Args:
        x: (B, C, H, W)
        window_size: window size
        h_w: Height of window
        w_w: Width of window
    Returns:
        local window features (num_windows*B, window_size*window_size, C)
    """
    B, C, H, W = x.shape
    x = x.view(B, C, H // window_size, window_size, W // window_size, window_size)
    windows = x.permute(0, 2, 4, 3, 5, 1).reshape(-1, window_size*window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: local window features (num_windows*B, window_size, window_size, C)
        window_size: Window size
        H: Height of image
        W: Width of image
    Returns:
        x: (B, C, H, W)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.reshape(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 5, 1, 3, 2, 4).reshape(B,windows.shape[2], H, W)
    return x


def _load_state_dict(module, state_dict, strict=False, logger=None):
    """Load state_dict to a module.

    This method is modified from :meth:`torch.nn.Module.load_state_dict`.
    Default value for ``strict`` is set to ``False`` and the message for
    param mismatch will be shown even if strict is False.

    Args:
        module (Module): Module that receives the state_dict.
        state_dict (OrderedDict): Weights.
        strict (bool): whether to strictly enforce that the keys
            in :attr:`state_dict` match the keys returned by this module's
            :meth:`~torch.nn.Module.state_dict` function. Default: ``False``.
        logger (:obj:`logging.Logger`, optional): Logger to log the error
            message. If not specified, print function will be used.
    """
    unexpected_keys = []
    all_missing_keys = []
    err_msg = []

    metadata = getattr(state_dict, '_metadata', None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata
    
    def load(module, prefix=''):
        local_metadata = {} if metadata is None else metadata.get(
            prefix[:-1], {})
        module._load_from_state_dict(state_dict, prefix, local_metadata, True,
                                     all_missing_keys, unexpected_keys,
                                     err_msg)
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + '.')

    load(module)
    load = None
    missing_keys = [
        key for key in all_missing_keys if 'num_batches_tracked' not in key
    ]

    if unexpected_keys:
        err_msg.append('unexpected key in source '
                       f'state_dict: {", ".join(unexpected_keys)}\n')
    if missing_keys:
        err_msg.append(
            f'missing keys in source state_dict: {", ".join(missing_keys)}\n')

    
    if len(err_msg) > 0:
        err_msg.insert(
            0, 'The model and loaded state dict do not match exactly\n')
        err_msg = '\n'.join(err_msg)
        if strict:
            raise RuntimeError(err_msg)
        elif logger is not None:
            logger.warning(err_msg)
        else:
            print(err_msg)


def _load_checkpoint(model,
                    filename,
                    map_location='cpu',
                    strict=False,
                    logger=None):
    """Load checkpoint from a file or URI.

    Args:
        model (Module): Module to load checkpoint.
        filename (str): Accept local filepath, URL, ``torchvision://xxx``,
            ``open-mmlab://xxx``. Please refer to ``docs/model_zoo.md`` for
            details.
        map_location (str): Same as :func:`torch.load`.
        strict (bool): Whether to allow different params for the model and
            checkpoint.
        logger (:mod:`logging.Logger` or None): The logger for error message.

    Returns:
        dict or OrderedDict: The loaded checkpoint.
    """
    checkpoint = torch.load(filename, map_location=map_location)
    if not isinstance(checkpoint, dict):
        raise RuntimeError(
            f'No state_dict found in checkpoint file {filename}')
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k[7:]: v for k, v in state_dict.items()}

    if sorted(list(state_dict.keys()))[0].startswith('encoder'):
        state_dict = {k.replace('encoder.', ''): v for k, v in state_dict.items() if k.startswith('encoder.')}

    _load_state_dict(model, state_dict, strict, logger)
    return checkpoint


class Downsample(nn.Module):
    """
    Down-sampling block"
    """

    def __init__(self,
                 dim,
                 keep_dim=False,
                 ):
        """
        Args:
            dim: feature size dimension.
            norm_layer: normalization layer.
            keep_dim: bool argument for maintaining the resolution.
        """

        super().__init__()
        if keep_dim:
            dim_out = dim
        else:
            dim_out = 2 * dim
        self.reduction = nn.Sequential(
            nn.Conv2d(dim, dim_out, 3, 2, 1, bias=False),
        )

    def forward(self, x):
        x = self.reduction(x)
        return x


class PatchEmbed(nn.Module):
    """
    Patch embedding block"
    """

    def __init__(self, in_chans=3, in_dim=64, dim=96):
        """
        Args:
            in_chans: number of input channels.
            dim: feature size dimension.
        """
        # in_dim = 1
        super().__init__()
        self.proj = nn.Identity()
        # Increase BatchNorm eps to prevent division by zero issues
        # eps=1e-4 is too small and can cause NaN when variance is near zero
        self.conv_down = nn.Sequential(
            nn.Conv2d(in_chans, in_dim, 3, 2, 1, bias=False),
            nn.BatchNorm2d(in_dim, eps=1e-3, momentum=0.01),
            nn.ReLU(),
            nn.Conv2d(in_dim, dim, 3, 2, 1, bias=False),
            nn.BatchNorm2d(dim, eps=1e-3, momentum=0.01),
            nn.ReLU()
            )

    def forward(self, x):
        x = self.proj(x)
        x = self.conv_down(x)
        return x


class ConvBlock(nn.Module):

    def __init__(self, dim,
                 drop_path=0.,
                 layer_scale=None,
                 kernel_size=3):
        super().__init__()

        self.conv1 = nn.Conv2d(dim, dim, kernel_size=kernel_size, stride=1, padding=1)
        self.norm1 = nn.BatchNorm2d(dim, eps=1e-5)
        self.act1 = nn.GELU(approximate= 'tanh')
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=kernel_size, stride=1, padding=1)
        self.norm2 = nn.BatchNorm2d(dim, eps=1e-5)
        self.layer_scale = layer_scale
        if layer_scale is not None and type(layer_scale) in [int, float]:
            self.gamma = nn.Parameter(layer_scale * torch.ones(dim))
            self.layer_scale = True
        else:
            self.layer_scale = False
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.norm2(x)
        if self.layer_scale:
            x = x * self.gamma.view(1, -1, 1, 1)
        x = input + self.drop_path(x)
        return x


class MambaVisionMixer(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=4,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        conv_bias=True,
        bias=False,
        use_fast_path=True, 
        layer_idx=None,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.use_fast_path = use_fast_path
        self.layer_idx = layer_idx
        self.in_proj = nn.Linear(self.d_model, self.d_inner, bias=bias, **factory_kwargs)    
        self.x_proj = nn.Linear(
            self.d_inner//2, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner//2, bias=True, **factory_kwargs)
        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError
        dt = torch.exp(
            torch.rand(self.d_inner//2, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        self.dt_proj.bias._no_reinit = True
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner//2,
        ).contiguous()
        A_log = torch.log(A)
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True
        self.D = nn.Parameter(torch.ones(self.d_inner//2, device=device))
        self.D._no_weight_decay = True
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.conv1d_x = nn.Conv1d(
            in_channels=self.d_inner//2,
            out_channels=self.d_inner//2,
            bias=conv_bias//2,
            kernel_size=d_conv,
            groups=self.d_inner//2,
            **factory_kwargs,
        )
        self.conv1d_z = nn.Conv1d(
            in_channels=self.d_inner//2,
            out_channels=self.d_inner//2,
            bias=conv_bias//2,
            kernel_size=d_conv,
            groups=self.d_inner//2,
            **factory_kwargs,
        )

    def forward(self, hidden_states):
        """
        hidden_states: (B, L, D)
        Returns: same shape as hidden_states
        """
        _, seqlen, _ = hidden_states.shape
        xz = self.in_proj(hidden_states)
        xz = rearrange(xz, "b l d -> b d l")
        x, z = xz.chunk(2, dim=1)
        A = -torch.exp(self.A_log.float())
        x = F.silu(F.conv1d(input=x, weight=self.conv1d_x.weight, bias=self.conv1d_x.bias, padding='same', groups=self.d_inner//2))
        z = F.silu(F.conv1d(input=z, weight=self.conv1d_z.weight, bias=self.conv1d_z.bias, padding='same', groups=self.d_inner//2))
        x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = rearrange(self.dt_proj(dt), "(b l) d -> b d l", l=seqlen)
        B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        y = selective_scan_fn(x, 
                              dt, 
                              A, 
                              B, 
                              C, 
                              self.D.float(), 
                              z=None, 
                              delta_bias=self.dt_proj.bias.float(), 
                              delta_softplus=True, 
                              return_last_state=None)
        
        y = torch.cat([y, z], dim=1)
        y = rearrange(y, "b d l -> b l d")
        out = self.out_proj(y)
        return out
    

class Attention(nn.Module):

    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=False,
            qk_norm=False,
            attn_drop=0.,
            proj_drop=0.,
            norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = True

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
             q, k, v,
                dropout_p=self.attn_drop.p,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(self, 
                 dim, 
                 num_heads, 
                 counter, 
                 transformer_blocks, 
                 mlp_ratio=4., 
                 qkv_bias=False, 
                 qk_scale=False, 
                 drop=0., 
                 attn_drop=0.,
                 drop_path=0., 
                 act_layer=nn.GELU, 
                 norm_layer=nn.LayerNorm, 
                 Mlp_block=Mlp,
                 layer_scale=None,
                 use_kan=False,
                 kan_grid_size=10,
                 kan_spline_order=2,
                 ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        if counter in transformer_blocks:
            self.mixer = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            norm_layer=norm_layer,
        )
        else:
            self.mixer = MambaVisionMixer(d_model=dim, 
                                          d_state=8,  
                                          d_conv=3,    
                                          expand=1
                                          )

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        
        # Conditional MLP/KAN selection
        if use_kan and _KAN_BACKEND is not None:
            if _KAN_BACKEND == "local_efficient_kan" and LocalEfficientKAN is not None:
                self.mlp = LocalEfficientKAN(
                    in_features=dim,
                    hidden_features=mlp_hidden_dim,
                    out_features=dim,
                    grid_size=kan_grid_size,
                    spline_order=kan_spline_order,
                    num_layers=2,
                    dropout=drop,
                )
            elif _KAN_BACKEND == "efficientkan" and KAN_MLP is not None:
                self.mlp = KAN_MLP(
                    in_features=dim,
                    hidden_features=mlp_hidden_dim,
                    out_features=dim,
                    grid_size=kan_grid_size,
                    spline_order=kan_spline_order,
                    dropout=drop,
                )
            elif _KAN_BACKEND == "efficient_kan":
                try:
                    self.mlp = EfficientKAN(
                        in_features=dim,
                        hidden_features=mlp_hidden_dim,
                        out_features=dim,
                        grid_size=kan_grid_size,
                        spline_order=kan_spline_order,
                        dropout=drop,
                    )
                except TypeError:
                    print("Warning: EfficientKAN API mismatch, falling back to standard MLP")
                    self.mlp = Mlp_block(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
            else:
                self.mlp = Mlp_block(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        else:
            if use_kan and _KAN_BACKEND is None:
                print("Warning: KAN requested but no KAN backend available. Using standard MLP.")
            self.mlp = Mlp_block(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
            
        use_layer_scale = layer_scale is not None and type(layer_scale) in [int, float]
        self.gamma_1 = nn.Parameter(layer_scale * torch.ones(dim))  if use_layer_scale else 1
        self.gamma_2 = nn.Parameter(layer_scale * torch.ones(dim))  if use_layer_scale else 1

    def forward(self, x):
        x = x + self.drop_path(self.gamma_1 * self.mixer(self.norm1(x)))
        x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x


class MambaVisionLayer(nn.Module):
    """
    MambaVision layer"
    """

    def __init__(self,
                 dim,
                 depth,
                 num_heads,
                 window_size,
                 conv=False,
                 downsample=True,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 layer_scale=None,
                 layer_scale_conv=None,
                 transformer_blocks = [],
                 use_kan=False,
                 kan_grid_size=5,
                 kan_spline_order=3,
    ):
        """
        Args:
            dim: feature size dimension.
            depth: number of layers in each stage.
            window_size: window size in each stage.
            conv: bool argument for conv stage flag.
            downsample: bool argument for down-sampling.
            mlp_ratio: MLP ratio.
            num_heads: number of heads in each stage.
            qkv_bias: bool argument for query, key, value learnable bias.
            qk_scale: bool argument to scaling query, key.
            drop: dropout rate.
            attn_drop: attention dropout rate.
            drop_path: drop path rate.
            norm_layer: normalization layer.
            layer_scale: layer scaling coefficient.
            layer_scale_conv: conv layer scaling coefficient.
            transformer_blocks: list of transformer blocks.
        """

        super().__init__()
        self.conv = conv
        self.transformer_block = False
        if conv:
            self.blocks = nn.ModuleList([ConvBlock(dim=dim,
                                                   drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                                   layer_scale=layer_scale_conv)
                                                   for i in range(depth)])
            self.transformer_block = False
        else:
            self.blocks = nn.ModuleList([Block(dim=dim,
                                               counter=i, 
                                               transformer_blocks=transformer_blocks,
                                               num_heads=num_heads,
                                               mlp_ratio=mlp_ratio,
                                               qkv_bias=qkv_bias,
                                               qk_scale=qk_scale,
                                               drop=drop,
                                               attn_drop=attn_drop,
                                               drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                               layer_scale=layer_scale,
                                               use_kan=use_kan,
                                               kan_grid_size=kan_grid_size,
                                               kan_spline_order=kan_spline_order)
                                               for i in range(depth)])
            self.transformer_block = True

        self.downsample = None if not downsample else Downsample(dim=dim)
        self.do_gt = False
        self.window_size = window_size

    def forward(self, x):
        _, _, H, W = x.shape

        if self.transformer_block:
            pad_r = (self.window_size - W % self.window_size) % self.window_size
            pad_b = (self.window_size - H % self.window_size) % self.window_size
            if pad_r > 0 or pad_b > 0:
                x = torch.nn.functional.pad(x, (0,pad_r,0,pad_b))
                _, _, Hp, Wp = x.shape
            else:
                Hp, Wp = H, W
            x = window_partition(x, self.window_size)

        for _, blk in enumerate(self.blocks):
            x = blk(x)
        if self.transformer_block:
            x = window_reverse(x, self.window_size, Hp, Wp)
            if pad_r > 0 or pad_b > 0:
                x = x[:, :, :H, :W].contiguous()
        if self.downsample is None:
            return x
        return self.downsample(x)


class MambaVision(nn.Module):
    """
    MambaVision,
    """

    def __init__(self,
                 dim,
                 in_dim,
                 depths,
                 window_size,
                 mlp_ratio,
                 num_heads,
                 drop_path_rate=0.2,
                 in_chans=3,
                 num_classes=1000,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 layer_scale=None,
                 layer_scale_conv=None,
                 use_hsi_adaptor=False,
                 hsi_bands=15,
                 use_kan=False,
                 kan_grid_size=10,
                 kan_spline_order=2,
                use_prompt=False,
                task_classes=6,
                prompt_inject_levels=[1, 2],  # Which levels to inject prompts (0-indexed)
                prompt_mode='full',  # 'full', 'visual_only', 'text_only' - for ablation study
                **kwargs):
        """
        Args:
            dim: feature size dimension.
            depths: number of layers in each stage.
            window_size: window size in each stage.
            mlp_ratio: MLP ratio.
            num_heads: number of heads in each stage.
            drop_path_rate: drop path rate.
            in_chans: number of input channels.
            num_classes: number of classes.
            qkv_bias: bool argument for query, key, value learnable bias.
            qk_scale: bool argument to scaling query, key.
            drop_rate: dropout rate.
            attn_drop_rate: attention dropout rate.
            norm_layer: normalization layer.
            layer_scale: layer scaling coefficient.
            layer_scale_conv: conv layer scaling coefficient.
        """
        super().__init__()
        num_features = int(dim * 2 ** (len(depths) - 1))
        # Debug: Print num_features calculation
        if len(depths) < 4:  # Only print for ablation cases
            print(f"Model initialization: depths={depths}, num_features={num_features}, dim={dim}")
        self.num_classes = num_classes
        # Store depths and dim for debugging
        self.depths = depths
        self.dim = dim
        # Prompt integration settings
        self.use_prompt = use_prompt
        self.prompt_mode = prompt_mode if use_prompt else None  # Store prompt_mode for ablation
        self.prompt_inject_levels = prompt_inject_levels if use_prompt else []
        # Optional hyperspectral adaptor (non-invasive): spectral 3D stem -> mean over spectral axis
        self.use_hsi_adaptor = use_hsi_adaptor
        if self.use_hsi_adaptor:
            # 3D spectral stem maps [B, 1, C', H, W] -> [B, dim, D, H, W]
            self.spectral_conv3d = nn.Sequential(
                nn.Conv3d(1, in_dim, kernel_size=(3,3,3), padding=(1,1,1), bias=False),
                nn.BatchNorm3d(in_dim, eps=1e-3, momentum=0.01),
                nn.SiLU(),
                nn.Conv3d(in_dim, dim, kernel_size=(3,3,3), padding=(1,1,1), bias=False),
                nn.BatchNorm3d(dim, eps=1e-3, momentum=0.01),
                nn.SiLU(),
            )
            # PatchEmbed consumes dim-channel 2D tensor
            self.patch_embed = PatchEmbed(in_chans=dim, in_dim=in_dim, dim=dim)
        else:
            self.patch_embed = PatchEmbed(in_chans=in_chans, in_dim=in_dim, dim=dim)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.levels = nn.ModuleList()
        for i in range(len(depths)):
            conv = True if (i == 0 or i == 1) else False
            level = MambaVisionLayer(dim=int(dim * 2 ** i),
                                     depth=depths[i],
                                     num_heads=num_heads[i],
                                     window_size=window_size[i],
                                     mlp_ratio=mlp_ratio,
                                     qkv_bias=qkv_bias,
                                     qk_scale=qk_scale,
                                     conv=conv,
                                     drop=drop_rate,
                                     attn_drop=attn_drop_rate,
                                     drop_path=dpr[sum(depths[:i]):sum(depths[:i + 1])],
                                     downsample=(i < len(depths) - 1),  # Downsample all stages except the last one
                                     layer_scale=layer_scale,
                                     layer_scale_conv=layer_scale_conv,
                                     transformer_blocks=list(range(depths[i]//2+1, depths[i])) if depths[i]%2!=0 else list(range(depths[i]//2, depths[i])),
                                     use_kan=use_kan,
                                     kan_grid_size=kan_grid_size,
                                     kan_spline_order=kan_spline_order,
                                     )
            self.levels.append(level)
        
        # Add prompt modules if enabled
        if self.use_prompt:
            # Initialize text prompt module
            self.text_prompt = Text_Prompt(task_classes=task_classes)
            self.clip_prompts = self.text_prompt.get_clip_prompt()
            
            # Initialize prompt and fusion modules for each injection level
            self.prompt_modules = nn.ModuleList()  # Use ModuleList to maintain order
            self.fusion_modules = nn.ModuleList()
            self.prompt_level_indices = []  # Store which level_idx each prompt corresponds to
            
            for level_idx in sorted(self.prompt_inject_levels):  # Sort for consistency
                if level_idx >= len(depths):
                    continue
                
                # Calculate channel dimensions AFTER the level processes (including downsample)
                # Each level receives dim*2**i as input, processes it, then downsamples
                # Level structure: i=0: dim->2*dim, i=1: 2*dim->4*dim, i=2: 4*dim->8*dim (for 4 stages)
                # We inject AFTER the level (after downsample), so we need OUTPUT channels
                # Last stage doesn't downsample, so check based on actual number of stages
                has_downsample = (level_idx < len(depths) - 1)  # All stages except the last one have downsample
                if has_downsample:
                    level_channels = int(dim * 2 ** (level_idx + 1))
                else:
                    level_channels = int(dim * 2 ** level_idx)
                
                # Estimate prompt size (after downsample, spatial size is smaller)
                # Input is typically downsampled by 2**(level_idx+1) by this point
                prompt_size = max(16, 64 // (2 ** (level_idx + 1)))
                
                # Create TVSP module for this level
                prompt_module = TVSP(
                    task_classes=task_classes,
                    prompt_size=prompt_size,
                    prompt_dim=level_channels,
                    out_dim=level_channels,
                    clip_prompts=self.clip_prompts
                )
                
                # Create PromptFusion module (input dim = feature + prompt = 2*level_channels)
                fusion_module = PromptFusion(
                    dim=level_channels * 2,  # feature + prompt concatenated
                    out_dim=level_channels,
                    head=num_heads[level_idx] if level_idx < len(num_heads) else 8,
                    ffn_expansion_factor=mlp_ratio,
                    bias=False
                )
                
                self.prompt_modules.append(prompt_module)
                self.fusion_modules.append(fusion_module)
                self.prompt_level_indices.append(level_idx)
            
            # Initialize prompt parameters with smaller scale to prevent NaN
            self._initialize_prompts_safely()
        else:
            self.text_prompt = None
            
        self.norm = nn.BatchNorm2d(num_features)
        # Debug: Verify BatchNorm was created with correct num_features
        if len(depths) < 4:
            print(f"BatchNorm created with num_features={num_features}, running_mean shape={self.norm.running_mean.shape}")
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, LayerNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm3d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
            # Initialize running stats to prevent division by zero
            if m.running_mean is not None:
                nn.init.zeros_(m.running_mean)
            if m.running_var is not None:
                nn.init.ones_(m.running_var)  # Initialize to 1.0, not 0.0

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'rpb'}
    
    def _initialize_prompts_safely(self):
        """
        Initialize prompt parameters with safe scaling to prevent gradient explosion and NaN.
        This is important when prompts are randomly initialized (not loaded from checkpoint).
        """
        if not self.use_prompt:
            return
        
        # Scale down visual prompts (large parameter tensors) - CRITICAL for stability
        # Even smaller initialization for fresh start without checkpoint
        for prompt_module in self.prompt_modules:
            if hasattr(prompt_module, 'visual_prompt'):
                # Initialize with VERY small scale (0.001 instead of default 1.0 from randn)
                # This prevents large activations that can lead to NaN
                # Fresh initialization should be more conservative than checkpoint loading
                with torch.no_grad():
                    prompt_module.visual_prompt.data *= 0.001  # 100x smaller than randn
            
            if hasattr(prompt_module, 'text_prompt_learnable'):
                # Scale down text prompt learnable parameters - even smaller for fresh init
                with torch.no_grad():
                    prompt_module.text_prompt_learnable.data *= 0.01  # 5x smaller
            
            # Also initialize conv_last and linear layers with smaller weights
            if hasattr(prompt_module, 'conv_last'):
                with torch.no_grad():
                    # Smaller initialization for final conv to prevent large outputs
                    nn.init.xavier_uniform_(prompt_module.conv_last.weight, gain=0.5)
            
            if hasattr(prompt_module, 'text_linear'):
                with torch.no_grad():
                    nn.init.xavier_uniform_(prompt_module.text_linear.weight, gain=0.5)
            
            if hasattr(prompt_module, 'clip_linear'):
                with torch.no_grad():
                    nn.init.xavier_uniform_(prompt_module.clip_linear.weight, gain=0.5)
            
            # Initialize cross_transformer components with smaller weights
            if hasattr(prompt_module, 'cross_transformer'):
                with torch.no_grad():
                    # Initialize temperature parameter (controls attention sharpness)
                    if hasattr(prompt_module.cross_transformer, 'attn'):
                        if hasattr(prompt_module.cross_transformer.attn, 'temperature'):
                            # Start with smaller temperature to avoid extreme attention values
                            prompt_module.cross_transformer.attn.temperature.data.clamp_(0.1, 1.0)
                    
                    # Initialize conv layers in cross_transformer with smaller gains
                    for name, module in prompt_module.cross_transformer.named_modules():
                        if isinstance(module, nn.Conv2d):
                            nn.init.xavier_uniform_(module.weight, gain=0.5)
                            if module.bias is not None:
                                nn.init.zeros_(module.bias)
    
    def get_prompt_parameters(self):
        """
        Get prompt-related parameters for setting different learning rates.
        This includes all adapters created dynamically.
        
        Returns:
            list: Parameters from prompt_modules, fusion_modules, text_prompt, and adapters
        """
        if not self.use_prompt:
            return []
        
        prompt_params = []
        # Collect all prompt module parameters
        # Use ModuleList iteration
        for module in self.prompt_modules:
            prompt_params.extend(list(module.parameters()))
        for module in self.fusion_modules:
            prompt_params.extend(list(module.parameters()))
        
        # Collect adapter parameters (created dynamically for dimension mismatch)
        for name, module in self.named_modules():
            if 'prompt_adapter' in name or 'fusion_adapter' in name:
                prompt_params.extend(list(module.parameters()))
        
        # Add text_prompt parameters (excluding frozen CLIP embeddings)
        if self.text_prompt is not None:
            # Text_Prompt doesn't have learnable params (CLIP is frozen)
            # But we include it for completeness
            prompt_params.extend([])  # Text_Prompt has no trainable params
        
        return prompt_params

    def _generate_prompt_ablation(self, prompt_module, x, text_prompt, prompt_weights, prompt_mode):
        """
        Generate prompt with ablation modes: visual_only, text_only, or full.
        
        Args:
            prompt_module: TVSP module
            x: Feature map [B, C, H, W]
            text_prompt: Text prompt from CLIP [B, 512]
            prompt_weights: One-hot task weights [B, task_classes]
            prompt_mode: 'full', 'visual_only', or 'text_only'
        
        Returns:
            prompt: Generated prompt [B, C, H, W]
        """
        B, C, H, W = x.shape
        
        if prompt_mode == 'visual_only':
            # Use only visual prompt, skip text prompt
            # Use visual prompt only by bypassing text prompt computation
            visual_prompt = prompt_module.visual_prompt.repeat(B, 1, 1, 1)  # [B, prompt_dim, prompt_size, prompt_size]
            
            # Interpolate to match feature size
            prompt = F.interpolate(visual_prompt, size=(H, W), mode='bilinear', align_corners=False)
            prompt = prompt_module.conv_last(prompt)
            
        elif prompt_mode == 'text_only':
            # Use only text prompt, skip visual prompt
            # Generate text prompt from CLIP
            if prompt_weights is None:
                # Default to first task
                prompt_weights = torch.zeros(B, prompt_module.task_classes, device=x.device)
                prompt_weights[:, 0] = 1.0
            
            # Compute text prompt (from TVSP logic)
            text_prompt_learnable = prompt_module.text_prompt_learnable  # [1, task_classes, prompt_dim, 1, 1]
            text_prompt_tensor = prompt_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * text_prompt_learnable.repeat(B, 1, 1, 1, 1)
            text_prompt_tensor = torch.mean(text_prompt_tensor, dim=1)  # [B, prompt_dim, 1, 1]
            
            # Apply CLIP linear transformation
            if text_prompt is not None:
                # text_prompt comes in as [B, 512]; TVSP.clip_linear expects last dim=512
                # so we apply it in 2D and then reshape back to [B, prompt_dim, 1, 1]
                text_prompt_processed = prompt_module.clip_linear(text_prompt)  # [B, prompt_dim]
                text_prompt_processed = text_prompt_processed.unsqueeze(-1).unsqueeze(-1)  # [B, prompt_dim, 1, 1]
                text_prompt_combined = text_prompt_tensor * text_prompt_processed
            else:
                # If no text_prompt provided, use learnable only
                text_prompt_combined = text_prompt_tensor
            
            # Interpolate to match feature size
            prompt = F.interpolate(text_prompt_combined, size=(H, W), mode='bilinear', align_corners=False)
            prompt = prompt_module.conv_last(prompt)
            
        else:  # 'full' - default behavior
            # Use full TVSP forward
            prompt = prompt_module(x, text_prompt, prompt_weights)
        
        return prompt

    def forward_features(self, x, task_id=None):
        # CRITICAL: Check input data for NaN/inf BEFORE any processing
        if torch.isnan(x).any() or torch.isinf(x).any():
            print(f"⚠️  CRITICAL: NaN/Inf detected in INPUT DATA!")
            print(f"   Input shape: {x.shape}")
            print(f"   NaN count: {torch.isnan(x).sum().item()}")
            print(f"   Inf count: {torch.isinf(x).sum().item()}")
            print(f"   Input stats: min={x.min().item():.4f}, max={x.max().item():.4f}, mean={x.mean().item():.4f}")
            # Replace with zeros as fallback
            x = torch.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)
            print(f"   ⚠️  Replaced NaN/Inf with finite values (may affect training quality)")
        
        if self.use_hsi_adaptor:
            # x: [B, C', H, W] -> [B, 1, C', H, W]
            if x.dim() != 4:
                raise ValueError("HSI mode expects input [B, C, H, W]")
            
            # CRITICAL: Clamp input values to prevent float16 overflow in spectral_conv3d
            # Float16 has range [-65504, 65504], but we clamp more conservatively
            x = x.clamp_(-10.0, 10.0)
            x = torch.nan_to_num(x, nan=0.0, posinf=10.0, neginf=-10.0)
            
            x = x.unsqueeze(1)
            x = self.spectral_conv3d(x)
            
            # Check for NaN after spectral conv and clamp if needed
            if torch.isnan(x).any() or torch.isinf(x).any():
                print(f"⚠️  CRITICAL: NaN/Inf detected after spectral_conv3d!")
                print(f"   Input stats before conv: min={x.min().item():.4f}, max={x.max().item():.4f}")
                x = torch.nan_to_num(x, nan=0.0, posinf=10.0, neginf=-10.0)
                x = x.clamp_(-10.0, 10.0)
            
            # average over spectral depth (dim=2)
            x = x.mean(dim=2)
            
            # Check for NaN after mean
            if torch.isnan(x).any() or torch.isinf(x).any():
                print(f"⚠️  CRITICAL: NaN/Inf detected after spectral mean!")
                x = torch.nan_to_num(x, nan=0.0, posinf=10.0, neginf=-10.0)
                x = x.clamp_(-10.0, 10.0)
        
        # Check before patch_embed
        if torch.isnan(x).any():
            print(f"⚠️  CRITICAL: NaN detected BEFORE patch_embed!")
            print(f"   Shape: {x.shape}, NaN count: {torch.isnan(x).sum().item()}")
            x = torch.nan_to_num(x, nan=0.0)
        
        x = self.patch_embed(x)
        
        # CRITICAL: Check for NaN immediately after patch_embed
        if torch.isnan(x).any():
            print(f"⚠️  CRITICAL: NaN detected AFTER patch_embed!")
            print(f"   Shape: {x.shape}, NaN count: {torch.isnan(x).sum().item()}")
            print(f"   This indicates patch_embed layer (BatchNorm or Conv) is causing NaN!")
            print(f"   Possible causes:")
            print(f"   1. BatchNorm running_stats not initialized (try model.train() first)")
            print(f"   2. Input values too large/small for BatchNorm")
            print(f"   3. KAN layers causing numerical instability (try --use-kan False)")
            x = torch.nan_to_num(x, nan=0.0)
        
        # Generate text prompts if using prompts
        if self.use_prompt:
            # If task_id is None, use default task (0 = Gaussian noise) or disable prompts
            if task_id is None:
                # For classification, we can't determine task, so use default or disable
                # Option: Use a default task_id for all samples
                task_id = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
            
            # Ensure task_id is tensor
            if not isinstance(task_id, torch.Tensor):
                task_id = torch.tensor([task_id] if isinstance(task_id, int) else task_id, dtype=torch.long, device=x.device)
            if task_id.dim() == 0:
                task_id = task_id.unsqueeze(0).expand(x.size(0))
            
            text_prompt, prompt_weights = self.text_prompt(x, task_id)
        else:
            text_prompt, prompt_weights = None, None
        
        # Process through levels with prompt injection
        prompt_idx = 0  # Index into prompt_modules ModuleList
        for level_idx, level in enumerate(self.levels):
            # Check for NaN BEFORE level processing (to catch base model issues)
            if torch.isnan(x).any():
                print(f"⚠️  CRITICAL: NaN detected BEFORE level {level_idx} processing!")
                print(f"   This indicates NaN is coming from base MambaVision model, not prompts!")
                print(f"   Shape: {x.shape}, NaN count: {torch.isnan(x).sum().item()}")
                x = torch.nan_to_num(x, nan=0.0)
            
            x = level(x)
            
            # Check for NaN AFTER level processing (before prompt injection)
            if torch.isnan(x).any():
                print(f"⚠️  CRITICAL: NaN detected AFTER level {level_idx} processing!")
                print(f"   This indicates NaN is coming from MambaVisionLayer, not prompts!")
                print(f"   Shape: {x.shape}, NaN count: {torch.isnan(x).sum().item()}")
                x = torch.nan_to_num(x, nan=0.0)
            
            # Inject prompt after this level if specified
            if self.use_prompt and level_idx in self.prompt_inject_levels:
                # Get the corresponding prompt and fusion modules
                if prompt_idx < len(self.prompt_modules):
                    prompt_module = self.prompt_modules[prompt_idx]
                    fusion_module = self.fusion_modules[prompt_idx]
                    
                    # Get actual channel dimension from feature map
                    actual_channels = x.size(1)
                    # Get expected prompt output channels from conv_last (TVSP uses this to output)
                    expected_prompt_channels = prompt_module.conv_last.out_channels
                    
                    # Check for NaN in input features before prompt generation
                    if torch.isnan(x).any():
                        print(f"⚠️  WARNING: NaN detected in features before prompt generation at level {level_idx}")
                        x = torch.nan_to_num(x, nan=0.0)
                    
                    # Generate prompt based on prompt_mode (for ablation study)
                    if hasattr(self, 'prompt_mode') and self.prompt_mode is not None and self.prompt_mode != 'full':
                        prompt = self._generate_prompt_ablation(prompt_module, x, text_prompt, prompt_weights, self.prompt_mode)
                    else:
                        # Default: full prompt system
                        prompt = prompt_module(x, text_prompt, prompt_weights)
                    
                    # Check for NaN in prompt output
                    if torch.isnan(prompt).any():
                        print(f"⚠️  WARNING: NaN detected in prompt output at level {level_idx}")
                        prompt = torch.nan_to_num(prompt, nan=0.0)
                    
                    # Handle dimension mismatch: adapt prompt to match feature channels
                    if actual_channels != prompt.size(1):
                        # Create adapter conv on-the-fly if needed - PROPERLY REGISTERED as module
                        adapter_key = f'prompt_adapter_{level_idx}'
                        if not hasattr(self, adapter_key):
                            adapter = nn.Conv2d(
                                prompt.size(1), actual_channels, 
                                kernel_size=1, bias=False
                            ).to(prompt.device)
                            # Use very small initialization to prevent NaN (gain=0.1 instead of default 1.0)
                            # This is critical when adapting between very different channel sizes
                            nn.init.xavier_uniform_(adapter.weight, gain=0.1)
                            self.add_module(adapter_key, adapter)  # Properly register as submodule
                            print(f"⚠️  Auto-adapting prompt channels at level {level_idx}: {prompt.size(1)} -> {actual_channels}")
                        adapter = getattr(self, adapter_key)
                        
                        # Check for NaN before and after adapter
                        if torch.isnan(prompt).any():
                            print(f"⚠️  WARNING: NaN detected in prompt before adapter at level {level_idx}")
                            prompt = torch.nan_to_num(prompt, nan=0.0)
                        
                        prompt = adapter(prompt)
                        
                        # Check for NaN after adapter
                        if torch.isnan(prompt).any():
                            print(f"⚠️  WARNING: NaN detected in prompt after adapter at level {level_idx}")
                            prompt = torch.nan_to_num(prompt, nan=0.0)
                    
                    # Now dimensions should match
                    assert actual_channels == prompt.size(1), \
                        f"Dimension mismatch at level {level_idx} after adaptation: feature={actual_channels}, prompt={prompt.size(1)}"
                    
                    # Verify and adapt PromptFusion if needed
                    expected_fusion_dim = actual_channels + prompt.size(1)  # x + prompt = 2 * actual_channels
                    fusion_transformer_dim = fusion_module.transformer.norm1.body.normalized_shape[0]
                    
                    if expected_fusion_dim != fusion_transformer_dim:
                        # Adapt fusion input dimension - PROPERLY REGISTERED as module
                        fusion_adapter_key = f'fusion_adapter_{level_idx}'
                        if not hasattr(self, fusion_adapter_key):
                            fusion_adapter = nn.Conv2d(
                                expected_fusion_dim, fusion_transformer_dim,
                                kernel_size=1, bias=False
                            ).to(x.device)
                            # Use very small initialization to prevent NaN (gain=0.1)
                            nn.init.xavier_uniform_(fusion_adapter.weight, gain=0.1)
                            self.add_module(fusion_adapter_key, fusion_adapter)  # Properly register as submodule
                            print(f"⚠️  Auto-adapting fusion input at level {level_idx}: {expected_fusion_dim} -> {fusion_transformer_dim}")
                        
                        fusion_adapter = getattr(self, fusion_adapter_key)
                        
                        # Concatenate and adapt, then manually call fusion components
                        # Check for NaN before concatenation
                        if torch.isnan(x).any() or torch.isnan(prompt).any():
                            print(f"⚠️  WARNING: NaN detected before fusion at level {level_idx}")
                            x = torch.nan_to_num(x, nan=0.0)
                            prompt = torch.nan_to_num(prompt, nan=0.0)
                        
                        fused_input = torch.cat([x, prompt], dim=1)
                        fused_input = fusion_adapter(fused_input)
                        
                        # Check for NaN after adapter
                        if torch.isnan(fused_input).any():
                            print(f"⚠️  WARNING: NaN detected after fusion adapter at level {level_idx}")
                            fused_input = torch.nan_to_num(fused_input, nan=0.0)
                        
                        # Apply transformer and conv manually
                        x = fusion_module.transformer(fused_input)
                        
                        # Check for NaN after transformer
                        if torch.isnan(x).any():
                            print(f"⚠️  WARNING: NaN detected after fusion transformer at level {level_idx}")
                            x = torch.nan_to_num(x, nan=0.0)
                        
                        x = fusion_module.conv(x)
                        
                        # Check for NaN after conv
                        if torch.isnan(x).any():
                            print(f"⚠️  WARNING: NaN detected after fusion conv at level {level_idx}")
                            x = torch.nan_to_num(x, nan=0.0)
                    else:
                        # Normal path: dimensions match perfectly
                        # Check for NaN before fusion
                        if torch.isnan(x).any() or torch.isnan(prompt).any():
                            print(f"⚠️  WARNING: NaN detected before normal fusion at level {level_idx}")
                            x = torch.nan_to_num(x, nan=0.0)
                            prompt = torch.nan_to_num(prompt, nan=0.0)
                        
                        x = fusion_module(x, prompt)
                        
                        # Check for NaN after fusion
                        if torch.isnan(x).any():
                            print(f"⚠️  WARNING: NaN detected after normal fusion at level {level_idx}")
                            x = torch.nan_to_num(x, nan=0.0)
                    prompt_idx += 1
        
        # Debug: Check feature dimensions before norm
        if hasattr(self, '_debug_printed') is False:
            print(f"Features before norm: shape={x.shape}, expected channels={self.norm.num_features}, BatchNorm running_mean shape={self.norm.running_mean.shape}")
            if x.shape[1] != self.norm.num_features:
                print(f"⚠️  MISMATCH: Features have {x.shape[1]} channels but BatchNorm expects {self.norm.num_features} channels!")
                print(f"   This will cause an error. The model architecture doesn't match the BatchNorm layer.")
            self._debug_printed = True
        
        # Verify dimensions match before applying norm
        if x.shape[1] != self.norm.num_features:
            raise RuntimeError(
                f"Feature dimension mismatch: input has {x.shape[1]} channels but BatchNorm expects {self.norm.num_features} channels. "
                f"This suggests the model was created with incorrect depths or num_features calculation. "
                f"Expected num_features based on depths: {int(self.dim * 2 ** (len(self.depths) - 1))}"
            )
        
        x = self.norm(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
        # Final NaN check before returning features
        if torch.isnan(x).any():
            print("⚠️  WARNING: NaN detected in final features, replacing with zeros")
            x = torch.nan_to_num(x, nan=0.0)
        
        return x

    def forward(self, x, task_id=None):
        """
        Args:
            x: Input tensor [B, C, H, W]
            task_id: Task ID tensor for prompt generation [B] or single value.
                    Required if use_prompt=True. None otherwise.
        
        ⚠️  IMPORTANT for Training Stability:
        - Enable gradient clipping in your training script (recommended: clip_grad=1.0 or 2.0)
        - Use a small learning rate for prompt parameters (e.g., 1e-5 to 1e-6)
        - Monitor for NaN warnings - they indicate numerical instability
        """
        # Check for NaN in input
        if torch.isnan(x).any():
            print("⚠️  WARNING: NaN detected in model input, replacing with zeros")
            x = torch.nan_to_num(x, nan=0.0)
        
        x = self.forward_features(x, task_id=task_id)
        x = self.head(x)
        
        # Final NaN check before returning output
        if torch.isnan(x).any():
            print("⚠️  WARNING: NaN detected in model output, replacing with zeros")
            x = torch.nan_to_num(x, nan=0.0)
        
        # Clamp output to reasonable range to prevent extreme values
        # This helps prevent NaN in loss computation (e.g., log(0) in CrossEntropy)
        x = torch.clamp(x, min=-100.0, max=100.0)
        
        return x

    def _load_state_dict(self, 
                         pretrained, 
                         strict: bool = False):
        _load_checkpoint(self, 
                         pretrained, 
                         strict=strict)
    
    def load_prompt_weights(self, checkpoint_path=None, strict=False, verbose=True, skip_on_shape_mismatch=True):
        """
        Load pretrained prompt weights from MP-HSIR checkpoint (OPTIONAL).
        
        ⚠️ IMPORTANT: Prompt blocks work WITHOUT checkpoint loading!
        - Without checkpoint: Prompts start with random initialization (trainable from scratch)
        - With checkpoint: Prompts load learned weights from MP-HSIR (can fine-tune or freeze)
        
        ⚠️ RECOMMENDED: Skip checkpoint loading if shape mismatches occur!
        - Partial loading (only 4/68 parameters) can cause instability
        - Fresh random initialization is often more stable
        
        This function loads ALL learnable parameters from the prompt blocks:
        - visual_prompt (learnable Parameter)
        - text_prompt_learnable (learnable Parameter, initialized from CLIP then trained)
        - cross_transformer.* (all weights in CrossTransformer)
        - conv_last.* (final conv layer)
        - clip_linear.* and text_linear.* (linear layers)
        - fusion transformer.* and conv.* (all PromptFusion weights)
        
        Args:
            checkpoint_path: Path to MP-HSIR checkpoint file or directory (optional).
                           - If None: tries to find checkpoints in common locations
                           - If file: loads that specific checkpoint
                           - If directory: automatically finds checkpoint files (.pth, .pth.tar, .ckpt, etc.)
                           Examples: 
                             - '/path/to/checkpoint.pth' (specific file)
                             - '/path/to/checkpoints' (directory - will auto-find latest)
                             - None (will search common locations)
            strict: Whether to strictly enforce matching keys
            verbose: Print detailed loading information
            
        Returns:
            Dictionary with loading statistics
        """
        if not self.use_prompt:
            print("Warning: use_prompt is False. Prompt modules not initialized.")
            return {}
        
        # If no path provided, try to find checkpoints in common locations
        if checkpoint_path is None:
            common_paths = [
                Path('checkpoints'),
                Path('./checkpoints'),
                Path('../checkpoints'),
                Path('ckpt'),
                Path('./ckpt'),
            ]
            
            found_paths = []
            for path in common_paths:
                if path.exists():
                    # Check if it's a file
                    if path.is_file():
                        checkpoint_path = path
                        break
                    # Check if it's a directory with checkpoint files
                    elif path.is_dir():
                        patterns = ['*.pth', '*.pth.tar', '*.ckpt', '*.pt']
                        for pattern in patterns:
                            files = list(path.glob(pattern))
                            if files:
                                found_paths.extend(files)
            
            # If we found files in directories, use the most recent one
            if checkpoint_path is None and found_paths:
                checkpoint_path = sorted(found_paths, key=lambda x: x.stat().st_mtime, reverse=True)[0]
                if verbose:
                    print(f"Auto-detected checkpoint: {checkpoint_path}")
            
            # If still None, show error
            if checkpoint_path is None:
                error_msg = (
                    f"❌ No checkpoint_path provided and none found in common locations.\n"
                    f"   Please provide checkpoint_path explicitly:\n"
                    f"   model.load_prompt_weights('checkpoints/Remote_sensing.ckpt')\n"
                    f"   or\n"
                    f"   model.load_prompt_weights('checkpoints')  # directory\n"
                    f"   \n"
                    f"   Searched in: {', '.join([str(p) for p in common_paths])}"
                )
                print(error_msg)
                return {}
        
        # Validate and resolve checkpoint path
        checkpoint_path = Path(checkpoint_path)
        
        # If path is a directory, try to find checkpoint files
        if checkpoint_path.is_dir():
            if verbose:
                print(f"Path is a directory: {checkpoint_path}")
                print("Searching for checkpoint files...")
            
            # Look for common checkpoint file patterns
            patterns = ['*.pth', '*.pth.tar', '*.ckpt', '*.pt', 'checkpoint*.pth', 'best*.pth']
            checkpoint_files = []
            for pattern in patterns:
                checkpoint_files.extend(checkpoint_path.glob(pattern))
                checkpoint_files.extend(checkpoint_path.glob(pattern.upper()))
            
            if not checkpoint_files:
                error_msg = (
                    f"❌ No checkpoint files found in directory: {checkpoint_path}\n"
                    f"   Searched for: {', '.join(patterns)}\n"
                    f"   Please provide the full path to a checkpoint file, e.g.:\n"
                    f"   '{checkpoint_path}/checkpoint.pth' or '{checkpoint_path}/best_model.pth'"
                )
                print(error_msg)
                return {}
            
            # Use the most recent checkpoint file, or first found
            checkpoint_file = sorted(checkpoint_files, key=lambda x: x.stat().st_mtime, reverse=True)[0]
            if verbose:
                print(f"✓ Found checkpoint file: {checkpoint_file}")
                if len(checkpoint_files) > 1:
                    print(f"   (Selected from {len(checkpoint_files)} found files)")
            checkpoint_path = checkpoint_file
        elif not checkpoint_path.exists():
            error_msg = (
                f"❌ Checkpoint path does not exist: {checkpoint_path}\n"
                f"   Please check the path and try again."
            )
            print(error_msg)
            return {}
        elif not checkpoint_path.is_file():
            error_msg = (
                f"❌ Path exists but is not a file: {checkpoint_path}\n"
                f"   If it's a directory, the code will search for checkpoint files in it."
            )
            print(error_msg)
            return {}
        
        # Load checkpoint
        try:
            checkpoint = torch.load(str(checkpoint_path), map_location='cpu')
        except Exception as e:
            error_msg = (
                f"❌ Failed to load checkpoint from: {checkpoint_path}\n"
                f"   Error: {e}\n"
                f"   Please verify the file is a valid PyTorch checkpoint."
            )
            print(error_msg)
            return {}
        
        state_dict = checkpoint.get('state_dict', checkpoint.get('model', checkpoint))
        
        # Remove common prefixes if present (module., net., model., etc.)
        if len(state_dict) > 0:
            first_key = list(state_dict.keys())[0]
            # Check and strip prefixes in order of commonality
            if first_key.startswith('module.'):
                state_dict = {k[7:]: v for k, v in state_dict.items()}
            elif first_key.startswith('net.'):
                state_dict = {k[4:]: v for k, v in state_dict.items()}
            elif first_key.startswith('model.'):
                state_dict = {k[6:]: v for k, v in state_dict.items()}
        
        # Filter prompt-related keys (this captures ALL nested sub-modules)
        prompt_state = {}
        prompt_keys_found = {
            'text_prompt': [],
            'prompt1': [],
            'prompt2': [],
            'fusion1': [],
            'fusion2': []
        }
        
        for key, value in state_dict.items():
            # Match text_prompt and all its sub-modules
            if key.startswith('text_prompt.'):
                prompt_state[key] = value
                prompt_keys_found['text_prompt'].append(key)
            # Match prompt1/prompt2 and map to ModuleList indices
            # prompt1 -> prompt_modules[0] if level 1 is first in prompt_inject_levels
            # prompt2 -> prompt_modules[1] if level 2 is second in prompt_inject_levels
            elif key.startswith('prompt1.'):
                # Map prompt1 to first ModuleList entry (assuming level 1 is first)
                # Find index of level 1 in prompt_level_indices
                module_idx = 0  # Default: assume prompt1 is first
                if hasattr(self, 'prompt_level_indices') and 1 in self.prompt_level_indices:
                    module_idx = self.prompt_level_indices.index(1)
                new_key = key.replace('prompt1.', f'prompt_modules.{module_idx}.')
                prompt_state[new_key] = value
                prompt_keys_found['prompt1'].append(key)
            elif key.startswith('prompt2.'):
                # Map prompt2 to second ModuleList entry (assuming level 2 is second)
                module_idx = 1  # Default: assume prompt2 is second
                if hasattr(self, 'prompt_level_indices') and 2 in self.prompt_level_indices:
                    module_idx = self.prompt_level_indices.index(2)
                new_key = key.replace('prompt2.', f'prompt_modules.{module_idx}.')
                prompt_state[new_key] = value
                prompt_keys_found['prompt2'].append(key)
            # Match fusion1/fusion2 similarly
            elif key.startswith('fusion1.'):
                module_idx = 0  # Default: assume fusion1 is first
                if hasattr(self, 'prompt_level_indices') and 1 in self.prompt_level_indices:
                    module_idx = self.prompt_level_indices.index(1)
                new_key = key.replace('fusion1.', f'fusion_modules.{module_idx}.')
                prompt_state[new_key] = value
                prompt_keys_found['fusion1'].append(key)
            elif key.startswith('fusion2.'):
                module_idx = 1  # Default: assume fusion2 is second
                if hasattr(self, 'prompt_level_indices') and 2 in self.prompt_level_indices:
                    module_idx = self.prompt_level_indices.index(2)
                new_key = key.replace('fusion2.', f'fusion_modules.{module_idx}.')
                prompt_state[new_key] = value
                prompt_keys_found['fusion2'].append(key)
        
        if len(prompt_state) == 0:
            print("Warning: No prompt weights found in checkpoint.")
            print(f"Available keys in checkpoint (first 10): {list(state_dict.keys())[:10]}")
            return {}
        
        if verbose:
            print(f"\n{'='*60}")
            print("Prompt Weight Loading Summary")
            print(f"{'='*60}")
            for category, keys in prompt_keys_found.items():
                if keys:
                    print(f"\n{category}: Found {len(keys)} parameters")
                    # Show key examples
                    example_keys = [k.split('.')[-2:] for k in keys[:3]]
                    print(f"  Examples: {['.'.join(k) for k in example_keys]}")
                    # Check for critical learnable parameters
                    if 'prompt1' in category or 'prompt2' in category:
                        has_visual = any('visual_prompt' in k for k in keys)
                        has_text = any('text_prompt_learnable' in k for k in keys)
                        has_cross = any('cross_transformer' in k for k in keys)
                        print(f"  ✓ visual_prompt: {has_visual}")
                        print(f"  ✓ text_prompt_learnable: {has_text}")
                        print(f"  ✓ cross_transformer: {has_cross}")
        
        # Load with strict=False to allow partial loading
        missing_keys, unexpected_keys = [], []
        try:
            model_dict = self.state_dict()
            filtered_dict = {}
            shape_mismatches = []
            
            for k, v in prompt_state.items():
                if k in model_dict:
                    if v.shape == model_dict[k].shape:
                        filtered_dict[k] = v
                    else:
                        shape_mismatches.append((k, v.shape, model_dict[k].shape))
                        if strict:
                            unexpected_keys.append(k)
                else:
                    if strict:
                        missing_keys.append(k)
            
            if shape_mismatches and verbose:
                print(f"\n⚠ Shape mismatches (will be skipped):")
                for k, ckpt_shape, model_shape in shape_mismatches[:5]:
                    print(f"  {k}: checkpoint{ckpt_shape} vs model{model_shape}")
                
                # Check if too many parameters failed to load
                load_ratio = len(filtered_dict) / len(prompt_state) if len(prompt_state) > 0 else 0
                if load_ratio < 0.5:  # Less than 50% loaded
                    print(f"\n{'='*60}")
                    print(f"⚠️  WARNING: Only {len(filtered_dict)}/{len(prompt_state)} parameters loaded ({load_ratio:.1%})")
                    print(f"⚠️  Too many shape mismatches detected!")
                    print(f"⚠️  RECOMMENDATION: Skip checkpoint loading and use fresh initialization!")
                    print(f"⚠️  Partial loading can cause instability and NaN issues!")
                    print(f"{'='*60}")
                    
                    if skip_on_shape_mismatch:
                        print(f"\n⚠️  Skipping checkpoint loading due to shape mismatches...")
                        print(f"⚠️  Using fresh random initialization instead (safer for stability)")
                        print(f"⚠️  Re-initializing prompts with safe small values...")
                        # Re-initialize prompts with safe values
                        self._initialize_prompts_safely()
                        return {
                            'loaded': 0,
                            'total': len(prompt_state),
                            'skipped': True,
                            'reason': 'Too many shape mismatches'
                        }
            
            model_dict.update(filtered_dict)
            self.load_state_dict(model_dict, strict=False)
            
            print(f"\n{'='*60}")
            print(f"✓ Successfully loaded {len(filtered_dict)}/{len(prompt_state)} prompt parameters")
            if len(filtered_dict) < len(prompt_state) * 0.5:
                print(f"⚠️  WARNING: Less than 50% of parameters loaded - may cause instability!")
            print(f"{'='*60}")
            
            if missing_keys and verbose:
                print(f"\n⚠ Missing keys (not in model): {len(missing_keys)}")
                if len(missing_keys) <= 5:
                    for k in missing_keys:
                        print(f"  - {k}")
                else:
                    for k in missing_keys[:5]:
                        print(f"  - {k}")
                    print(f"  ... and {len(missing_keys)-5} more")
            
            return {
                'loaded': len(filtered_dict),
                'total': len(prompt_state),
                'missing': missing_keys,
                'unexpected': unexpected_keys,
                'shape_mismatches': shape_mismatches
            }
        except Exception as e:
            print(f"\n❌ Error loading prompt weights: {e}")
            import traceback
            traceback.print_exc()
            return {}
    
    def list_prompt_parameters(self):
        """
        List all learnable prompt parameters in the model.
        Useful for verifying what will be loaded from checkpoint.
        
        Returns:
            Dictionary with parameter names and shapes
        """
        if not self.use_prompt:
            print("Prompt modules not enabled.")
            return {}
        
        prompt_params = {}
        model_state = self.state_dict()
        
        for key in sorted(model_state.keys()):
            if 'prompt_modules' in key or 'fusion_modules' in key or 'text_prompt' in key:
                prompt_params[key] = model_state[key].shape
        
        print(f"\n{'='*60}")
        print(f"Learnable Prompt Parameters in Model ({len(prompt_params)} total)")
        print(f"{'='*60}")
        
        categories = {}
        for key in prompt_params.keys():
            if 'prompt_modules.prompt_1' in key:
                categories.setdefault('Prompt Level 1', []).append(key)
            elif 'prompt_modules.prompt_2' in key:
                categories.setdefault('Prompt Level 2', []).append(key)
            elif 'fusion_modules.fusion_1' in key:
                categories.setdefault('Fusion Level 1', []).append(key)
            elif 'fusion_modules.fusion_2' in key:
                categories.setdefault('Fusion Level 2', []).append(key)
            elif 'text_prompt' in key:
                categories.setdefault('Text Prompt', []).append(key)
        
        for cat, keys in categories.items():
            print(f"\n{cat} ({len(keys)} parameters):")
            for key in keys[:10]:  # Show first 10
                print(f"  {key}: {prompt_params[key]}")
            if len(keys) > 10:
                print(f"  ... and {len(keys)-10} more")
        
        return prompt_params


@register_pip_model
@register_model
def mamba_vision_T(pretrained=False, **kwargs):
    """
    MambaVision-T with optional prompt integration.
    
    Prompt Usage Examples:
        
        # Option 1: Use prompts WITHOUT checkpoint (train from scratch)
        model = mamba_vision_T(
            use_prompt=True,
            task_classes=6,  # or 7 for haze task
            prompt_inject_levels=[1, 2]  # Inject at levels 1 and 2
        )
        # Prompts are randomly initialized, ready to train
        task_id = torch.tensor([0])  # Task 0: Gaussian noise
        output = model(x, task_id=task_id)
        
        # Option 2: Use prompts WITH checkpoint (load learned weights)
        model = mamba_vision_T(use_prompt=True, task_classes=6)
        model.load_prompt_weights('path/to/mp_hsir_checkpoint.pth')  # OPTIONAL
        # Now prompts have learned weights from MP-HSIR
        output = model(x, task_id=task_id)
    """
    model_path = kwargs.pop("model_path", "/tmp/mamba_vision_T.pth.tar")
    # Handle depths: if None is explicitly passed, use default; otherwise use provided value
    depths = kwargs.pop("depths", None)
    if depths is None:
        depths = [1, 3, 8, 4]  # Default depths for mamba_vision_T
    num_heads = kwargs.pop("num_heads", [2, 4, 8, 16])
    window_size = kwargs.pop("window_size", [8, 8, 14, 7])
    dim = kwargs.pop("dim", 80)
    in_dim = kwargs.pop("in_dim", 32)
    mlp_ratio = kwargs.pop("mlp_ratio", 4)
    resolution = kwargs.pop("resolution", 224)
    drop_path_rate = kwargs.pop("drop_path_rate", 0.2)
    use_kan = kwargs.pop("use_kan", False)
    kan_grid_size = kwargs.pop("kan_grid_size", 5)
    kan_spline_order = kwargs.pop("kan_spline_order", 3)
    # Prompt-related kwargs (will be passed through)
    use_prompt = kwargs.pop("use_prompt", False)
    task_classes = kwargs.pop("task_classes", 6)
    prompt_inject_levels = kwargs.pop("prompt_inject_levels", [1, 2])
    prompt_mode = kwargs.pop("prompt_mode", "full")  # Add prompt_mode support
    pretrained_cfg = resolve_pretrained_cfg('mamba_vision_T').to_dict()
    update_args(pretrained_cfg, kwargs, kwargs_filter=None)
    
    # Adjust num_heads and window_size to match depths length if custom depths provided
    if len(depths) < len(num_heads):
        num_heads = num_heads[:len(depths)]
        window_size = window_size[:len(depths)]
    
    model = MambaVision(depths=depths,
                        num_heads=num_heads,
                        window_size=window_size,
                        dim=dim,
                        in_dim=in_dim,
                        mlp_ratio=mlp_ratio,
                        resolution=resolution,
                        drop_path_rate=drop_path_rate,
                        use_kan=use_kan,
                        kan_grid_size=kan_grid_size,
                        kan_spline_order=kan_spline_order,
                        use_prompt=use_prompt,
                        task_classes=task_classes,
                        prompt_inject_levels=prompt_inject_levels,
                        prompt_mode=prompt_mode,
                        **kwargs)
    model.pretrained_cfg = pretrained_cfg
    model.default_cfg = model.pretrained_cfg
    if pretrained:
        if not Path(model_path).is_file():
            url = model.default_cfg['url']
            torch.hub.download_url_to_file(url=url, dst=model_path)
        model._load_state_dict(model_path)
    return model


@register_pip_model
@register_model
def mamba_vision_T2(pretrained=False, **kwargs):
    model_path = kwargs.pop("model_path", "/tmp/mamba_vision_T2.pth.tar")
    depths = kwargs.pop("depths", [1, 3, 11, 4])
    num_heads = kwargs.pop("num_heads", [2, 4, 8, 16])
    window_size = kwargs.pop("window_size", [8, 8, 14, 7])
    dim = kwargs.pop("dim", 80)
    in_dim = kwargs.pop("in_dim", 32)
    mlp_ratio = kwargs.pop("mlp_ratio", 4)
    resolution = kwargs.pop("resolution", 224)
    drop_path_rate = kwargs.pop("drop_path_rate", 0.2)
    use_kan = kwargs.pop("use_kan", False)
    pretrained_cfg = resolve_pretrained_cfg('mamba_vision_T2').to_dict()
    update_args(pretrained_cfg, kwargs, kwargs_filter=None)
    model = MambaVision(depths=depths,
                        num_heads=num_heads,
                        window_size=window_size,
                        dim=dim,
                        in_dim=in_dim,
                        mlp_ratio=mlp_ratio,
                        resolution=resolution,
                        drop_path_rate=drop_path_rate,
                        use_kan=use_kan,
                        **kwargs)
    model.pretrained_cfg = pretrained_cfg
    model.default_cfg = model.pretrained_cfg
    if pretrained:
        if not Path(model_path).is_file():
            url = model.default_cfg['url']
            torch.hub.download_url_to_file(url=url, dst=model_path)
        model._load_state_dict(model_path)
    return model


@register_pip_model
@register_model
def mamba_vision_S(pretrained=False, **kwargs):
    model_path = kwargs.pop("model_path", "/tmp/mamba_vision_S.pth.tar")
    depths = kwargs.pop("depths", [3, 3, 7, 5])  # Allow custom depths for stage ablation
    num_heads = kwargs.pop("num_heads", [2, 4, 8, 16])
    window_size = kwargs.pop("window_size", [8, 8, 14, 7])
    dim = kwargs.pop("dim", 96)
    in_dim = kwargs.pop("in_dim", 64)
    mlp_ratio = kwargs.pop("mlp_ratio", 4)
    resolution = kwargs.pop("resolution", 224)
    drop_path_rate = kwargs.pop("drop_path_rate", 0.2)
    use_kan = kwargs.pop("use_kan", False)
    # Prompt-related kwargs
    prompt_mode = kwargs.pop("prompt_mode", "full")
    pretrained_cfg = resolve_pretrained_cfg('mamba_vision_S').to_dict()
    update_args(pretrained_cfg, kwargs, kwargs_filter=None)
    
    # Adjust num_heads and window_size to match depths length if custom depths provided
    if len(depths) < len(num_heads):
        num_heads = num_heads[:len(depths)]
        window_size = window_size[:len(depths)]
    
    model = MambaVision(depths=depths,
                        num_heads=num_heads,
                        window_size=window_size,
                        dim=dim,
                        in_dim=in_dim,
                        mlp_ratio=mlp_ratio,
                        resolution=resolution,
                        drop_path_rate=drop_path_rate,
                        use_kan=use_kan,
                        prompt_mode=prompt_mode,
                        **kwargs)
    model.pretrained_cfg = pretrained_cfg
    model.default_cfg = model.pretrained_cfg
    if pretrained:
        if not Path(model_path).is_file():
            url = model.default_cfg['url']
            torch.hub.download_url_to_file(url=url, dst=model_path)
        model._load_state_dict(model_path)
    return model


@register_pip_model
@register_model
def mamba_vision_B(pretrained=False, **kwargs):
    model_path = kwargs.pop("model_path", "/tmp/mamba_vision_B.pth.tar")
    depths = kwargs.pop("depths", [3, 3, 10, 5])  # Allow custom depths for stage ablation
    num_heads = kwargs.pop("num_heads", [2, 4, 8, 16])
    window_size = kwargs.pop("window_size", [8, 8, 14, 7])
    dim = kwargs.pop("dim", 128)
    in_dim = kwargs.pop("in_dim", 64)
    mlp_ratio = kwargs.pop("mlp_ratio", 4)
    resolution = kwargs.pop("resolution", 224)
    drop_path_rate = kwargs.pop("drop_path_rate", 0.3)
    layer_scale = kwargs.pop("layer_scale", 1e-5)
    use_kan = kwargs.pop("use_kan", False)
    # Prompt-related kwargs
    prompt_mode = kwargs.pop("prompt_mode", "full")
    pretrained_cfg = resolve_pretrained_cfg('mamba_vision_B').to_dict()
    update_args(pretrained_cfg, kwargs, kwargs_filter=None)
    
    # Adjust num_heads and window_size to match depths length if custom depths provided
    if len(depths) < len(num_heads):
        num_heads = num_heads[:len(depths)]
        window_size = window_size[:len(depths)]
    
    model = MambaVision(depths=depths,
                        num_heads=num_heads,
                        window_size=window_size,
                        dim=dim,
                        in_dim=in_dim,
                        mlp_ratio=mlp_ratio,
                        resolution=resolution,
                        drop_path_rate=drop_path_rate,
                        layer_scale=layer_scale,
                        layer_scale_conv=None,
                        prompt_mode=prompt_mode,
                        **kwargs)
    model.pretrained_cfg = pretrained_cfg
    model.default_cfg = model.pretrained_cfg
    if pretrained:
        if not Path(model_path).is_file():
            url = model.default_cfg['url']
            torch.hub.download_url_to_file(url=url, dst=model_path)
        model._load_state_dict(model_path)
    return model


@register_pip_model
@register_model
def mamba_vision_B_21k(pretrained=False, **kwargs):
    model_path = kwargs.pop("model_path", "/tmp/mamba_vision_B_21k.pth.tar")
    depths = kwargs.pop("depths", [3, 3, 10, 5])
    num_heads = kwargs.pop("num_heads", [2, 4, 8, 16])
    window_size = kwargs.pop("window_size", [8, 8, 14, 7])
    dim = kwargs.pop("dim", 128)
    in_dim = kwargs.pop("in_dim", 64)
    mlp_ratio = kwargs.pop("mlp_ratio", 4)
    resolution = kwargs.pop("resolution", 224)
    drop_path_rate = kwargs.pop("drop_path_rate", 0.3)
    layer_scale = kwargs.pop("layer_scale", 1e-5)
    use_kan = kwargs.pop("use_kan", False)
    pretrained_cfg = resolve_pretrained_cfg('mamba_vision_B_21k').to_dict()
    update_args(pretrained_cfg, kwargs, kwargs_filter=None)
    model = MambaVision(depths=depths,
                        num_heads=num_heads,
                        window_size=window_size,
                        dim=dim,
                        in_dim=in_dim,
                        mlp_ratio=mlp_ratio,
                        resolution=resolution,
                        drop_path_rate=drop_path_rate,
                        layer_scale=layer_scale,
                        layer_scale_conv=None,
                        **kwargs)
    model.pretrained_cfg = pretrained_cfg
    model.default_cfg = model.pretrained_cfg
    if pretrained:
        if not Path(model_path).is_file():
            url = model.default_cfg['url']
            torch.hub.download_url_to_file(url=url, dst=model_path)
        model._load_state_dict(model_path)
    return model


@register_pip_model
@register_model
def mamba_vision_L(pretrained=False, **kwargs):
    model_path = kwargs.pop("model_path", "/tmp/mamba_vision_L.pth.tar")
    depths = kwargs.pop("depths", [3, 3, 10, 5])
    num_heads = kwargs.pop("num_heads", [4, 8, 16, 32])
    window_size = kwargs.pop("window_size", [8, 8, 14, 7])
    dim = kwargs.pop("dim", 196)
    in_dim = kwargs.pop("in_dim", 64)
    mlp_ratio = kwargs.pop("mlp_ratio", 4)
    resolution = kwargs.pop("resolution", 224)
    drop_path_rate = kwargs.pop("drop_path_rate", 0.3)
    layer_scale = kwargs.pop("layer_scale", 1e-5)
    use_kan = kwargs.pop("use_kan", False)
    pretrained_cfg = resolve_pretrained_cfg('mamba_vision_L').to_dict()
    update_args(pretrained_cfg, kwargs, kwargs_filter=None)
    model = MambaVision(depths=depths,
                        num_heads=num_heads,
                        window_size=window_size,
                        dim=dim,
                        in_dim=in_dim,
                        mlp_ratio=mlp_ratio,
                        resolution=resolution,
                        drop_path_rate=drop_path_rate,
                        layer_scale=layer_scale,
                        layer_scale_conv=None,
                        **kwargs)
    model.pretrained_cfg = pretrained_cfg
    model.default_cfg = model.pretrained_cfg
    if pretrained:
        if not Path(model_path).is_file():
            url = model.default_cfg['url']
            torch.hub.download_url_to_file(url=url, dst=model_path)
        model._load_state_dict(model_path)
    return model


@register_pip_model
@register_model
def mamba_vision_L_21k(pretrained=False, **kwargs):
    model_path = kwargs.pop("model_path", "/tmp/mamba_vision_L_21k.pth.tar")
    depths = kwargs.pop("depths", [3, 3, 10, 5])
    num_heads = kwargs.pop("num_heads", [4, 8, 16, 32])
    window_size = kwargs.pop("window_size", [8, 8, 14, 7])
    dim = kwargs.pop("dim", 196)
    in_dim = kwargs.pop("in_dim", 64)
    mlp_ratio = kwargs.pop("mlp_ratio", 4)
    resolution = kwargs.pop("resolution", 224)
    drop_path_rate = kwargs.pop("drop_path_rate", 0.3)
    layer_scale = kwargs.pop("layer_scale", 1e-5)
    use_kan = kwargs.pop("use_kan", False)
    pretrained_cfg = resolve_pretrained_cfg('mamba_vision_L_21k').to_dict()
    update_args(pretrained_cfg, kwargs, kwargs_filter=None)
    model = MambaVision(depths=depths,
                        num_heads=num_heads,
                        window_size=window_size,
                        dim=dim,
                        in_dim=in_dim,
                        mlp_ratio=mlp_ratio,
                        resolution=resolution,
                        drop_path_rate=drop_path_rate,
                        layer_scale=layer_scale,
                        layer_scale_conv=None,
                        **kwargs)
    model.pretrained_cfg = pretrained_cfg
    model.default_cfg = model.pretrained_cfg
    if pretrained:
        if not Path(model_path).is_file():
            url = model.default_cfg['url']
            torch.hub.download_url_to_file(url=url, dst=model_path)
        model._load_state_dict(model_path)
    return model


@register_pip_model
@register_model
def mamba_vision_L2(pretrained=False, **kwargs):
    model_path = kwargs.pop("model_path", "/tmp/mamba_vision_L2.pth.tar")
    depths = kwargs.pop("depths", [3, 3, 12, 5])
    num_heads = kwargs.pop("num_heads", [4, 8, 16, 32])
    window_size = kwargs.pop("window_size", [8, 8, 14, 7])
    dim = kwargs.pop("dim", 196)
    in_dim = kwargs.pop("in_dim", 64)
    mlp_ratio = kwargs.pop("mlp_ratio", 4)
    resolution = kwargs.pop("resolution", 224)
    drop_path_rate = kwargs.pop("drop_path_rate", 0.3)
    layer_scale = kwargs.pop("layer_scale", 1e-5)
    use_kan = kwargs.pop("use_kan", False)
    pretrained_cfg = resolve_pretrained_cfg('mamba_vision_L2').to_dict()
    update_args(pretrained_cfg, kwargs, kwargs_filter=None)
    model = MambaVision(depths=depths,
                        num_heads=num_heads,
                        window_size=window_size,
                        dim=dim,
                        in_dim=in_dim,
                        mlp_ratio=mlp_ratio,
                        resolution=resolution,
                        drop_path_rate=drop_path_rate,
                        layer_scale=layer_scale,
                        layer_scale_conv=None,
                        **kwargs)
    model.pretrained_cfg = pretrained_cfg
    model.default_cfg = model.pretrained_cfg
    if pretrained:
        if not Path(model_path).is_file():
            url = model.default_cfg['url']
            torch.hub.download_url_to_file(url=url, dst=model_path)
        model._load_state_dict(model_path)
    return model


@register_pip_model
@register_model
def mamba_vision_L2_512_21k(pretrained=False, **kwargs):
    model_path = kwargs.pop("model_path", "/tmp/mamba_vision_L2_512_21k.pth.tar")
    depths = kwargs.pop("depths", [3, 3, 12, 5])
    num_heads = kwargs.pop("num_heads", [4, 8, 16, 32])
    window_size = kwargs.pop("window_size", [8, 8, 32, 16])
    dim = kwargs.pop("dim", 196)
    in_dim = kwargs.pop("in_dim", 64)
    mlp_ratio = kwargs.pop("mlp_ratio", 4)
    resolution = kwargs.pop("resolution", 512)
    drop_path_rate = kwargs.pop("drop_path_rate", 0.3)
    layer_scale = kwargs.pop("layer_scale", 1e-5)
    use_kan = kwargs.pop("use_kan", False)
    pretrained_cfg = resolve_pretrained_cfg('mamba_vision_L2_512_21k').to_dict()
    update_args(pretrained_cfg, kwargs, kwargs_filter=None)
    model = MambaVision(depths=depths,
                        num_heads=num_heads,
                        window_size=window_size,
                        dim=dim,
                        in_dim=in_dim,
                        mlp_ratio=mlp_ratio,
                        resolution=resolution,
                        drop_path_rate=drop_path_rate,
                        layer_scale=layer_scale,
                        layer_scale_conv=None,
                        **kwargs)
    model.pretrained_cfg = pretrained_cfg
    model.default_cfg = model.pretrained_cfg
    if pretrained:
        if not Path(model_path).is_file():
            url = model.default_cfg['url']
            torch.hub.download_url_to_file(url=url, dst=model_path)
        model._load_state_dict(model_path)
    return model


@register_pip_model
@register_model
def mamba_vision_L3_256_21k(pretrained=False, **kwargs):
    model_path = kwargs.pop("model_path", "/tmp/mamba_vision_L3_256_21k.pth.tar")
    depths = kwargs.pop("depths", [3, 3, 20, 10])
    num_heads = kwargs.pop("num_heads", [4, 8, 16, 32])
    window_size = kwargs.pop("window_size", [8, 8, 16, 8])
    dim = kwargs.pop("dim", 256)
    in_dim = kwargs.pop("in_dim", 64)
    mlp_ratio = kwargs.pop("mlp_ratio", 4)
    resolution = kwargs.pop("resolution", 256)
    drop_path_rate = kwargs.pop("drop_path_rate", 0.5)
    layer_scale = kwargs.pop("layer_scale", 1e-5)
    use_kan = kwargs.pop("use_kan", False)
    pretrained_cfg = resolve_pretrained_cfg('mamba_vision_L3_256_21k').to_dict()
    update_args(pretrained_cfg, kwargs, kwargs_filter=None)
    model = MambaVision(depths=depths,
                        num_heads=num_heads,
                        window_size=window_size,
                        dim=dim,
                        in_dim=in_dim,
                        mlp_ratio=mlp_ratio,
                        resolution=resolution,
                        drop_path_rate=drop_path_rate,
                        layer_scale=layer_scale,
                        layer_scale_conv=None,
                        **kwargs)
    model.pretrained_cfg = pretrained_cfg
    model.default_cfg = model.pretrained_cfg
    if pretrained:
        if not Path(model_path).is_file():
            url = model.default_cfg['url']
            torch.hub.download_url_to_file(url=url, dst=model_path)
        model._load_state_dict(model_path)
    return model


@register_pip_model
@register_model
def mamba_vision_L3_512_21k(pretrained=False, **kwargs):
    model_path = kwargs.pop("model_path", "/tmp/mamba_vision_L3_512_21k.pth.tar")
    depths = kwargs.pop("depths", [3, 3, 20, 10])
    num_heads = kwargs.pop("num_heads", [4, 8, 16, 32])
    window_size = kwargs.pop("window_size", [8, 8, 32, 16])
    dim = kwargs.pop("dim", 256)
    in_dim = kwargs.pop("in_dim", 64)
    mlp_ratio = kwargs.pop("mlp_ratio", 4)
    resolution = kwargs.pop("resolution", 512)
    drop_path_rate = kwargs.pop("drop_path_rate", 0.5)
    layer_scale = kwargs.pop("layer_scale", 1e-5)
    use_kan = kwargs.pop("use_kan", False)
    pretrained_cfg = resolve_pretrained_cfg('mamba_vision_L3_512_21k').to_dict()
    update_args(pretrained_cfg, kwargs, kwargs_filter=None)
    model = MambaVision(depths=depths,
                        num_heads=num_heads,
                        window_size=window_size,
                        dim=dim,
                        in_dim=in_dim,
                        mlp_ratio=mlp_ratio,
                        resolution=resolution,
                        drop_path_rate=drop_path_rate,
                        layer_scale=layer_scale,
                        layer_scale_conv=None,
                        **kwargs)
    model.pretrained_cfg = pretrained_cfg
    model.default_cfg = model.pretrained_cfg
    if pretrained:
        if not Path(model_path).is_file():
            url = model.default_cfg['url']
            torch.hub.download_url_to_file(url=url, dst=model_path)
        model._load_state_dict(model_path)
    return model
