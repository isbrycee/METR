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
"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn

import numpy as np
from util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou


KPS_OKS_SIGMAS = np.array([
    .26, .25, .25, .35, .35, .79, .79, .72, .72, .62, .62, 1.07, 1.07,
    .87, .87, .89, .89
]) / 10.0


def joint_oks(src_joints, tgt_joints, tgt_bboxes, joint_sigmas=KPS_OKS_SIGMAS, with_center=True, eps=1e-15):
    tgt_flags = tgt_joints[:, :, 2]
    tgt_joints = tgt_joints[:, :, 0:2]
    tgt_wh = tgt_bboxes[..., 2:]
    tgt_areas = tgt_wh[..., 0] * tgt_wh[..., 1]
    num_gts, num_kpts = tgt_joints.shape[0:2]

    areas = tgt_areas.unsqueeze(1).expand(num_gts, num_kpts)
    sigmas = torch.tensor(joint_sigmas).type_as(tgt_joints)
    sigmas_sq = torch.square(2 * sigmas).unsqueeze(0).expand(num_gts, num_kpts)
    d_sq = torch.square(src_joints.unsqueeze(1) - tgt_joints.unsqueeze(0)).sum(-1)
    tgt_flags = tgt_flags.unsqueeze(0).expand(*d_sq.shape)

    oks = torch.exp(-d_sq / (2 * areas * sigmas_sq + eps))
    oks = oks * tgt_flags
    oks = oks.sum(-1) / (tgt_flags.sum(-1) + eps)

    return oks


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, args):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.class_normalization = args.set_class_normalization
        self.box_normalization = args.set_box_normalization

    def forward(self, outputs, targets, weight_dict, num_box):
        """ Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates

            match_args

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        with torch.no_grad():
            bs, num_queries = outputs["pred_logits"].shape[:2]
            # how to normalize loss for both class, box and keypoint
            NORMALIZER = {"num_box": num_box, "mean": num_queries, "none": 1, "box_average": num_box}
            # We flatten to compute the cost matrices in a batch
            out_logit = outputs["pred_logits"].flatten(0, 1)
            out_prob = out_logit.sigmoid()
            out_bbox = outputs["pred_boxes"].flatten(0, 1)

            # Also concat the target labels and boxes
            tgt_bbox = torch.cat([v["boxes"] for v in targets])
            sizes = [t["boxes"].shape[0] for t in targets]
            num_local = sum(sizes)

            if num_local == 0:
                return [(torch.as_tensor([], dtype=torch.int64), torch.as_tensor([], dtype=torch.int64)) for _ in sizes]

            assert ("loss_bce" in weight_dict) ^ ("loss_ce" in weight_dict)
            # Compute the classification cost.
            if "loss_bce" in weight_dict: # no
                cost_class = - out_prob * weight_dict["loss_bce"]
            elif "loss_ce" in weight_dict:
                alpha = 0.25
                gamma = 2.0
                neg_cost_class = (1 - alpha) * (out_prob ** gamma) * (-(1 - out_prob + 1e-8).log())
                pos_cost_class = alpha * ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())
                cost_class = pos_cost_class - neg_cost_class
                cost_class = cost_class * weight_dict["loss_ce"]
            cost_class = cost_class[..., None].repeat(1, num_local)

            C = cost_class / NORMALIZER[self.class_normalization]

            # Compute the L1 cost between boxes
            cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1) / NORMALIZER[self.box_normalization]  # 2000, 8

            # Compute the giou cost betwen boxes
            cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox),
                                            box_cxcywh_to_xyxy(tgt_bbox)) / NORMALIZER[self.box_normalization]
            # Final cost matrix
            C_box = weight_dict["loss_bbox"] * cost_bbox + weight_dict["loss_giou"] * cost_giou
            C = C + C_box

            C = C.view(bs, num_queries, -1).cpu()
            indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
            return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


def build_unified_matcher(args):
    return HungarianMatcher(args)
