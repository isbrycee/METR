# ------------------------------------------------------------------------
# Obj2Seq: Formatting Objects as Sequences with Class Prompt for Visual Tasks
# Copyright (c) 2022 CASIA & Sensetime. All Rights Reserved.
# ------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F

from util.misc import get_world_size, is_dist_avail_and_initialized
from .asl_losses import AsymmetricLoss, AsymmetricLossOptimized

def build_asymmetricloss(args):
    lossClass = AsymmetricLossOptimized if args.prompt_indicator_asl_optimized else AsymmetricLoss
    return lossClass(gamma_neg=args.prompt_indicator_asl_gamma_neg,
                     gamma_pos=args.prompt_indicator_asl_gamma_pos,
                     clip=args.prompt_indicator_asl_clip,
                     disable_torch_grad_focal_loss=True)

@torch.no_grad()
def accuracy(output, target):
        """Computes the precision@k for the specified values of k"""
        num_gt = target.size(0)
        _, pred = output.topk(num_gt, 1, True, True)
        union = set(pred[0].cpu().detach().numpy().tolist()) & set(target.cpu().detach().numpy().tolist())
        res = (len(union) / num_gt) * 100 if num_gt else 100
        return torch.tensor(res).cuda()
    
class ClassDecoderCriterion(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.losses = args.prompt_indicator_losses
        self.asl_loss = build_asymmetricloss(args)
        self.loss_funcs = {
            "asl": lambda outputs, targets: self.asl_loss(outputs['cls_label_logits'], targets["multi_label_onehot"], targets["multi_label_weights"]),
            "bce": lambda outputs, targets: F.binary_cross_entropy_with_logits(outputs['cls_label_logits'], targets["multi_label_onehot"], targets["multi_label_weights"], reduction="sum") / targets["multi_label_weights"].sum(),
        }

    def prepare_targets(self, outputs, targets):
        return {
            "multi_label_onehot": torch.stack([t["multi_label_onehot"] for t in targets], dim=0),
            "multi_label_weights": torch.stack([t["multi_label_weights"] for t in targets], dim=0),
        }

    def forward(self, outputs, aux_outputs, targets):
        targets = self.prepare_targets(outputs, targets)
        loss_dict = {}
        for loss in self.losses:
            loss_dict[f"cls_{loss}"] = self.loss_funcs[loss](outputs, targets)
            for i, aux_label_output in enumerate(aux_outputs):
                loss_dict[f"cls_{loss}_{i}"] = self.loss_funcs[loss](aux_label_output, targets)
        
        loss_dict['asl_class_error'] = 100 - accuracy(outputs['cls_label_logits'], targets["multi_label_onehot"].nonzero()[:,1])
        return loss_dict