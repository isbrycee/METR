# ------------------------------------------------------------------------
# Obj2Seq: Formatting Objects as Sequences with Class Prompt for Visual Tasks
# Copyright (c) 2022 CASIA & Sensetime. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Anchor DETR (https://github.com/megvii-research/AnchorDETR)
# Copyright (c) 2021 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
import torch
import torch.nn.functional as F
from torch import nn

from .unified_matcher import build_unified_matcher
from .asl_losses import sigmoid_focal_loss
from util import box_ops
from util.misc import (nested_tensor_from_tensor_list, interpolate,
                       get_world_size, is_dist_avail_and_initialized)
import numpy as np

class UnifiedSingleClassCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self, args, weight_dict={}):
        """ Create the criterion.
        Parameters:
            args.MATCHER: module able to compute a matching between targets and proposals
            args.focal_alpha: dict containing as key the names of the losses and as values their relative weight.
            args.*_loss_coef
            args.*_normalization
        """
        super().__init__()
        self.matcher = build_unified_matcher(args)
        self.focal_alpha = args.focal_alpha
        # weight dict
        self.all_weight_dict = weight_dict
        self.class_normalization = args.class_normalization
        self.box_normalization = args.box_normalization

    def loss_labels(self, outputs, targets, indices):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']
        idx = self._get_src_permutation_idx(indices)
        target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1]],
                                            dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
        target_classes_onehot[idx] = 1

        loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_boxes=None, alpha=self.focal_alpha, gamma=2) / self.loss_normalization[self.class_normalization]
        losses = {'loss_ce': loss_ce}

        if False:#log:
            # TODO this should probably be a separate loss, not hacked in this one here
            # TODO Fix here
            losses['class_error'] = 100 # - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    def loss_boxes(self, outputs, targets, indices):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, h, w), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / self.loss_normalization[self.box_normalization]

        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / self.loss_normalization[self.box_normalization]
        return losses
    
    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        assert 'pred_logits' in outputs
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices):
        loss_map = {
            'labels': self.loss_labels,
            'boxes': self.loss_boxes
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices)

    def build_weight_dict(self, losses):
        weight_dict = {}
        if "labels" in losses:
            weight_dict["loss_ce"] = self.all_weight_dict["loss_ce"]
        if "boxes" in losses:
            weight_dict["loss_bbox"] = self.all_weight_dict["loss_bbox"]
            weight_dict["loss_giou"] = self.all_weight_dict["loss_giou"]
        return weight_dict

    def forward(self, outputs, targets, losses, num_boxes, indices=None):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
                  pred_logits: bs, nobj
                  pred_boxes:  bs, nobj, 4
                  (optional):  bs, nobj, mngts
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
                  keypoints: ngts, 17, 3
        """
        weight_dict = self.build_weight_dict(losses)
        # normalize term
        self.loss_normalization = {"num_box": num_boxes, "mean": outputs["pred_logits"].shape[1], "none": 1}

        # Retrieve the matching between the outputs of the last layer and the targets
        if indices is None:
            indices = self.matcher(outputs, targets, weight_dict, num_boxes)

        loss_dict = {}
        for loss in losses:
            loss_dict.update(self.get_loss(loss, outputs, targets, indices))
            
        return loss_dict

    def rescale_loss(self, loss_dict, weight_dict):
        print('loss_dict, weight_dict', loss_dict, weight_dict)
        return {
            k: loss_dict[k] * weight_dict[k]
            for k in loss_dict.keys() if k in weight_dict
        }
