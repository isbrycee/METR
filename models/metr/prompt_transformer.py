import math
import copy
from typing import Optional, List

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from .attention import MultiheadAttention

class BasicDecoderLayer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.d_model = args.hidden_dim
        self.n_heads = args.nheads
        self.normalize_before = args.pre_norm

        # cross attention
        self.build_cross_attn(args)
        self.dropout1 = nn.Dropout(args.dropout)
        self.norm1 = nn.LayerNorm(self.d_model)

        # self attention
        self.self_attn = not args.prompt_indicator_no_self_attn
        if self.self_attn:
            self.self_attn = nn.MultiheadAttention(self.d_model, self.n_heads, dropout=args.dropout)
            self.dropout2 = nn.Dropout(args.self_attn_dropout)
            self.norm2 = nn.LayerNorm(self.d_model)

        # ffn
        self.ffn = FFN(self.d_model, args.dim_feedforward, args.dropout, args.activation, normalize_before=self.normalize_before)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def self_attn_forward(self, tgt, query_pos, **kwargs):
        if query_pos is not None and query_pos.shape[0] != tgt.shape[0]:
            cs = tgt.shape[0] // query_pos.shape[0]
            query_pos_self = query_pos.repeat_interleave(repeats=cs, dim=0)
        else:
            query_pos_self = query_pos
        q = k = self.with_pos_embed(tgt, query_pos_self)
        tgt2 = self.self_attn(q.transpose(0, 1), k.transpose(0, 1), tgt.transpose(0, 1))[0].transpose(0, 1)
        return tgt2

    def forward_post(self, tgt, query_pos, **kwargs): # here
        # self attention
        if self.self_attn:
            tgt2 = self.self_attn_forward(tgt, query_pos, **kwargs)
            tgt = tgt + self.dropout2(tgt2)
            tgt = self.norm2(tgt)
            
        tgt2 = self.cross_attn_forward(tgt, query_pos, **kwargs)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # ffn
        tgt = self.ffn(tgt)

        return tgt

    def forward_pre(self, tgt, query_pos, **kwargs):
        # self attention
        if self.self_attn:
            tgt2 = self.norm2(tgt)
            tgt2 = self.self_attn_forward(tgt2, query_pos, **kwargs)
            tgt = tgt + self.dropout2(tgt2)

        tgt2 = self.norm1(tgt)
        tgt2 = self.cross_attn_forward(tgt2, query_pos, **kwargs)
        tgt = tgt + self.dropout1(tgt2)

        # ffn
        tgt = self.ffn(tgt)

        return tgt

    def forward(self, *args, **kwargs):
        if self.normalize_before:
            return self.forward_pre(*args, **kwargs)
        return self.forward_post(*args, **kwargs)


class prompt_TransformerDecoderLayer(BasicDecoderLayer):
    def build_cross_attn(self, args):
        self.cross_attn = nn.MultiheadAttention(self.d_model, self.n_heads, dropout=args.dropout)

    def cross_attn_forward(self, tgt, query_pos, **kwargs):
        
        bs_all, seq, c = tgt.shape  
        srcs = kwargs["srcs"]
        bs = srcs.shape[1]
        
        if bs_all > bs: # what this mean?
            tgt = tgt.view(bs, -1, c)
            cs = bs_all // bs

        src_padding_masks = kwargs.pop("src_padding_masks")
        posemb_2d = kwargs.pop("posemb_2d", 0)
        # query_pos is None
        query_pos = torch.zeros_like(tgt) if query_pos is None else query_pos.repeat(1,cs,1)
        
        srcs = srcs.transpose(0,1)
        tgt2 = self.cross_attn((tgt + query_pos).transpose(0, 1),
                                (srcs + posemb_2d).reshape(bs, -1, c).transpose(0,1),
                                srcs.reshape(bs, -1, c).transpose(0, 1), key_padding_mask=src_padding_masks.reshape(bs, -1))[0].transpose(0,1)

        return tgt2.reshape(bs_all, seq, c)

    def forward(self, tgt, query_pos, reference_points, srcs, src_padding_masks, **kwargs):
        return super().forward(tgt, query_pos, srcs=srcs, src_padding_masks=src_padding_masks, **kwargs)

class FFN(nn.Module):

    def __init__(self, d_model=256, d_ffn=1024, dropout=0., activation='relu', normalize_before=False):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.normalize_before = normalize_before

    def forward_post(self, src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src

    def forward_pre(self, src):
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src2))))
        src = src + self.dropout3(src2)
        return src

    def forward(self, src):
        if self.normalize_before:
            return self.forward_pre(src)
        return self.forward_post(src)

    
def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    if activation == "prelu":
        return nn.PReLU()
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")