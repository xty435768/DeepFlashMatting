import torch
import math
import copy
from torch import nn
from torch import Tensor
from typing import Optional
import torch.nn.functional as F
from torch.nn.modules.container import ModuleList
from einops import rearrange
from model_new.cbam import CBAM

def _get_clones(module, N):
    return ModuleList([copy.deepcopy(module) for i in range(N)])

def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise ValueError("activation should be relu/gelu, not {}".format(activation))

class PatchEmbedding(nn.Module):
    def __init__(self, patch_size=16, in_channels=3, embed_dim=768):
        super(PatchEmbedding, self).__init__()
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x, target_h_shape, target_w_shape):
        b, c, h, w = x.shape
        target_h = target_h_shape * self.patch_size
        target_w = target_w_shape * self.patch_size
        pad_h = target_h - h
        pad_w = target_w - w
        pad_up = pad_h // 2
        pad_down = pad_h - pad_up
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        x = nn.ReflectionPad2d(padding=(pad_left, pad_right, pad_up, pad_down))(x)
        return self.proj(x).flatten(2).permute(2, 0, 1)


class PositionalEncoding(nn.Module):
    """
    Learnable position embeddings

    Args:
        pe_type (str): type of position embeddings,
            which is chosen from ['fully_learnable', 'sinusoidal']
        d_model (int): embed dim (required).
        max_len (int): max. length of the incoming sequence (default=5000).

    Examples:
        pos_encoder = PositionalEncoding(d_model, max_len=100)
    """

    def __init__(self, pe_type: str, d_model: int, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()

        if pe_type == "fully_learnable":
            self.pe = nn.parameter.Parameter(torch.randn(max_len, 1, d_model))
        elif pe_type == "sinusoidal":
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(
                torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
            )
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            pe = pe.unsqueeze(0).transpose(0, 1)
            self.register_buffer("pe", pe)
        else:
            raise RuntimeError(
                "PE type should be fully_learnable/sinusoidal, not {}".format(pe_type)
            )

    def forward(self, x: Tensor) -> Tensor:
        """

        Args:
            x (Tensor): the sequence fed to the positional encoder model [L, N, C]

        Returns:
            output (Tensor): position embeddings [L, N, C]

        """

        return x + self.pe[: x.size(0)]


class SelfAttentionTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead=4, dim_feedforward=1024, dropout=0.1, activation="relu", require_flatten=False, in_feature_dim=256):
        super(SelfAttentionTransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.require_flatten = require_flatten
        if self.require_flatten:
            self.flatten_linear = nn.Conv2d(in_feature_dim, d_model, kernel_size=1)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout_self = nn.Dropout(dropout)
        self.norm_self = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if "activation" not in state:
            state["activation"] = F.relu
        super(SelfAttentionTransformerEncoderLayer, self).__setstate__(state)

    def forward(
            self,
            src: Tensor,
            src_: Tensor = None,
            external_feature: Tensor = None,
            src_mask: Optional[Tensor] = None,
            src_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        if self.require_flatten:
            src = self.flatten_linear(src).flatten(start_dim=2).permute(2, 0, 1)
        src2 = self.self_attn(
            src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask
        )[0]
        src = src + self.dropout_self(src2)
        if external_feature is not None:
            src = src + external_feature
        src = self.norm_self(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


class SelfCoAttentionTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead=4, dim_feedforward=1024, dropout=0.1, activation="relu", require_flatten=False, in_feature_dim=256):
        super(SelfCoAttentionTransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.require_flatten = require_flatten
        if self.require_flatten:
            self.flatten_linear = nn.Conv2d(in_feature_dim, d_model, kernel_size=1)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout_self = nn.Dropout(dropout)
        self.norm_self = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if "activation" not in state:
            state["activation"] = F.relu
        super(SelfCoAttentionTransformerEncoderLayer, self).__setstate__(state)

    def forward(
            self,
            src: Tensor,
            src_: Tensor,
            external_feature: Tensor = None,
            src_mask: Optional[Tensor] = None,
            src_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        if self.require_flatten:
            src = self.flatten_linear(src).flatten(start_dim=2).permute(2, 0, 1)

        if src_ is not None:
            src2 = self.cross_attn(
                src, src_, src_, attn_mask=src_mask, key_padding_mask=src_key_padding_mask
            )[0]
            src = src + self.dropout1(src2)
        src2 = self.self_attn(
            src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask
        )[0]
        src = src + self.dropout_self(src2)
        src = self.norm_self(src)

        if external_feature is not None:
            src = src + external_feature
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


class TransformerEncoder(nn.Module):
    __constants__ = ["norm"]

    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerEncoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(
            self,
            src: Tensor,
            src_: Tensor = None,
            external_feature: Tensor = None,
            mask: Optional[Tensor] = None,
            src_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        output = src

        for mod in self.layers:
            output = mod(
                output,
                src_,
                external_feature,
                src_mask=mask,
                src_key_padding_mask=src_key_padding_mask,
            )

        if self.norm is not None:
            output = self.norm(output)

        return output

class FCM(nn.Module):
    def __init__(
            self,
            feature_dim: int,
            encoder_dim: int,
            image_size: int = 14,
            nhead: int = 4,
            nf_transformer_layer_num: int = 2,
            f_transformer_layer_num: int = 1,
    ):
        super(FCM, self).__init__()
        self.linear = nn.Conv2d(feature_dim, encoder_dim, kernel_size=1)
        self.patch_embed = PatchEmbedding(in_channels=3, embed_dim=encoder_dim)
        self.positional_encoding = PositionalEncoding(
            "fully_learnable", encoder_dim, image_size ** 2
        )

        co_encoder_layer = SelfCoAttentionTransformerEncoderLayer(
            d_model=encoder_dim, nhead=nhead, dim_feedforward=4 * encoder_dim
        )
        self_encoder_layer = SelfAttentionTransformerEncoderLayer(
            d_model=encoder_dim, nhead=nhead, dim_feedforward=4 * encoder_dim
        )
        self.noflash_transformer = TransformerEncoder(
            co_encoder_layer, num_layers=nf_transformer_layer_num
        )
        self.flash_transformer = TransformerEncoder(
            self_encoder_layer, num_layers=f_transformer_layer_num
        )
        self.channel_att = CBAM(encoder_dim)

    def forward(self, flash_features: Tensor, noflash_features: Tensor,
                flash_ratio: Tensor, noflash_ratio: Tensor):
        b, c, h, w = noflash_features.shape

        flash_features = self.linear(flash_features).flatten(start_dim=2).permute(2, 0, 1)
        noflash_features = self.linear(noflash_features).flatten(start_dim=2).permute(2, 0, 1)


        flash_features = self.positional_encoding(flash_features)
        noflash_features = self.positional_encoding(noflash_features)

        flash_features = self.flash_transformer(flash_features)
        noflash_features = self.noflash_transformer(noflash_features, flash_features)
        
        noflash_features = rearrange(noflash_features, '(h w) b c -> b c h w', h=h, w=w)
        
        return self.channel_att(noflash_features)