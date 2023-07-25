import torch

import numpy as np
from scipy import interpolate

from mmcv.runner import get_dist_info
import torch.nn as nn

def rpe_index(window_size):
    num_relative_distance = (2 * window_size[0] - 1) * (2 * window_size[1] - 1) + 3

    # get pair-wise relative position index for each token inside the window
    coords_h = torch.arange(window_size[0])
    coords_w = torch.arange(window_size[1])
    coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
    coords_flatten = torch.flatten(coords, 1)
    relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
    relative_coords = relative_coords.permute(1, 2, 0).contiguous()
    relative_coords[:, :, 0] += window_size[0] - 1  # shift to start from 0
    relative_coords[:, :, 1] += window_size[1] - 1
    relative_coords[:, :, 0] *= 2 * window_size[1] - 1
    relative_position_index = \
        torch.zeros(size=(window_size[0] * window_size[1] + 1, ) * 2, dtype=relative_coords.dtype)
    relative_position_index[1:, 1:] = relative_coords.sum(-1)
    relative_position_index[0, 0:] = num_relative_distance - 3
    relative_position_index[0:, 0] = num_relative_distance - 2
    relative_position_index[0, 0] = num_relative_distance - 1

    return relative_position_index

def prepare_rpe(rel_pos_bias, src_patch_shape, dst_patch_shape):
    src_num_pos, num_attn_heads = rel_pos_bias.size()
    rank, _ = get_dist_info()
    dst_num_pos = (dst_patch_shape[0]*2 -1) * (dst_patch_shape[1]*2 -1) + 3 

    if dst_patch_shape[0] != src_patch_shape[0] or dst_patch_shape[1] != src_patch_shape[1]:

        num_extra_tokens = dst_num_pos - (dst_patch_shape[0] * 2 - 1) * (dst_patch_shape[1] * 2 - 1)

        src_size_0, src_size_1 = src_patch_shape[0] * 2 - 1, src_patch_shape[1]*2 -1
        extra_tokens = rel_pos_bias[-num_extra_tokens:, :]
        rel_pos_bias = rel_pos_bias[:-num_extra_tokens, :]

        dst_size_0 = dst_patch_shape[0] * 2 - 1
        dst_size_1 = dst_patch_shape[1] * 2 - 1

        dim = rel_pos_bias.shape[-1]
        rel_pos_bias = rel_pos_bias.reshape(1 , src_size_0, src_size_1, dim).permute(0, 3, 1, 2)
        new_rel_pos_bias = nn.functional.interpolate(rel_pos_bias, scale_factor=(dst_size_0 / src_size_0, dst_size_1 / dst_size_1), mode='bicubic',) 
        new_rel_pos_bias = new_rel_pos_bias.permute(0, 2, 3, 1).view(1, -1, dim).squeeze(0)
        new_rel_pos_bias = torch.cat((new_rel_pos_bias, extra_tokens), dim=0)
    else:
        new_rel_pos_bias = rel_pos_bias
    relative_position_index = rpe_index(dst_patch_shape)

    new_rel_pos_bias = new_rel_pos_bias[relative_position_index.view(-1)].view(
                    dst_patch_shape[0] * dst_patch_shape[1] + 1,
                    dst_patch_shape[0] * dst_patch_shape[1] + 1, -1)
    new_rel_pos_bias = new_rel_pos_bias.permute(2, 0, 1).contiguous()
    return new_rel_pos_bias
