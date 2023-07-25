# ------------------------------------------------------------------------
# Obj2Seq: Formatting Objects as Sequences with Class Prompt for Visual Tasks
# Copyright (c) 2022 CASIA & Sensetime. All Rights Reserved.
# ------------------------------------------------------------------------
import torch
from torch import nn

from util.misc import (get_world_size, is_dist_avail_and_initialized)
from .unified_single_class_criterion import UnifiedSingleClassCriterion
from .secriterion_for_CEM import SetCriterion
import torch, os
from scipy.optimize import linear_sum_assignment
from torch import nn

from util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network
    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1, focal_alpha = 0.25, Topk = 20):
        """Creates the matcher
        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.Topk = Topk
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"

        self.focal_alpha = focal_alpha

    @torch.no_grad()
    def forward(self, outputs, targets):
        """ Performs the matching
        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates
            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates
        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """

        bs, num_queries = outputs["pred_logits"].shape[:2]

        # We flatten to compute the cost matrices in a batch
        out_prob = outputs["pred_logits"].flatten(0, 1).sigmoid()  # [batch_size * num_queries, num_classes]
        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]

        # Also concat the target labels and boxes
        tgt_ids = []
        for i,v in enumerate(targets):
            for _ in range(v["boxes"].shape[0]):
                tgt_ids.append(i % self.Topk)
        tgt_ids = torch.tensor(tgt_ids)
        tgt_bbox = torch.cat([v["boxes"] for v in targets])

        # Compute the classification cost.
        alpha = self.focal_alpha
        gamma = 2.0
        neg_cost_class = (1 - alpha) * (out_prob ** gamma) * (-(1 - out_prob + 1e-8).log())
        pos_cost_class = alpha * ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())
        cost_class = pos_cost_class[:, tgt_ids] - neg_cost_class[:, tgt_ids]

        # Compute the L1 cost between boxes
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)
            
        # Compute the giou cost betwen boxes            
        cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))

        # Final cost matrix
        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        C = C.view(bs, num_queries, -1).cpu()

        sizes = []
        sizes_temp = [t["boxes"].shape[0] for t in targets]
        for i in range(len(targets) // self.Topk):
            sizes.append(sum(sizes_temp[self.Topk * i:self.Topk * (i+1)]))
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]
    

class ClasswiseCriterion(nn.Module):
    def __init__(self, args, weight_dict = {}):
        super().__init__()
        self.set_criterion = UnifiedSingleClassCriterion(args, weight_dict)
        self.weight_dict = weight_dict
        self.use_cdn = args.use_dn
        self.num_classes_for_CEM = args.num_classes_for_CEM
        self.num_classes = args.retention_policy_train_max_classes
        self.num_decoder_layers=args.dec_layers
        if self.num_classes_for_CEM:
            self.CEM_matcher = HungarianMatcher(cost_class=args.set_cost_class,
                                                              cost_bbox=args.set_cost_bbox,
                                                              cost_giou=args.set_cost_giou,
                                                              Topk=args.train_topk,
                                                              focal_alpha=args.focal_alpha)
            self.CEM_set_criterion = SetCriterion(self.num_classes, 
                                                                self.CEM_matcher, 
                                                                weight_dict=weight_dict, 
                                                                focal_alpha=args.focal_alpha, 
                                                                losses=['labels', 'boxes'], 
                                                                args=args)

    def forward(self, output, targets, target_for_class):
        targets_temp = targets
        targets = target_for_class
        device = output['batch_index'].device
        cs_all = output['pred_logits'].shape[0]
        num_boxes = self.get_num_boxes(targets, device)
        bs_idx, cls_idx = output["batch_index"], output["class_index"] # cs_all
        task_info = {
            'losses': ['labels', 'boxes'],
            'required_targets': {'boxes': [0, 4]}
        }
        target = []
        for id_b, id_c in zip(bs_idx, cls_idx):
            tgtThis = {}
            id_c = id_c.item()
            if id_c in targets[id_b]:
                tgtOrigin = targets[id_b][id_c]
                for key in task_info['required_targets']:
                    tgtThis[key] = tgtOrigin[key].to(device)
            else:
                for key in task_info['required_targets']:
                    default_shape = task_info['required_targets'][key]
                    tgtThis[key] = torch.zeros(default_shape, device=device)
            target.append(tgtThis)
        losses = {}
        
        if self.num_classes_for_CEM == 20:
            output_interm = {
                "pred_logits": output['interm_outputs']["pred_logits"],
                "pred_boxes": output['interm_outputs']["pred_boxes"],
            }
            l_dict = self.CEM_set_criterion(output_interm, target, num_boxes)
            l_dict = {k + '_interm': v for k, v in l_dict.items()}
            losses.update(l_dict)
            
        # the first five layer in decoder, aux_outputs
        for i in range(len(output['aux_outputs'])):
            output_aux = {
                "pred_logits": output['aux_outputs'][i]["pred_logits"],
                "pred_boxes": output['aux_outputs'][i]["pred_boxes"],
            }
            l_dict = self.set_criterion(output_aux, target, task_info['losses'], num_boxes)
            l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
            losses.update(l_dict)
            
        # the sixth layer in decoder  
        output_six = {
            "pred_logits": output["pred_logits"],
            "pred_boxes": output["pred_boxes"]
        }
        l_dict = self.set_criterion(output_six, target, task_info['losses'], num_boxes)
        losses.update(l_dict)
        
        # dn
        dn_pos_idx = []
        dn_neg_idx = []
        dn_meta = output['dn_meta']
        if self.use_cdn:
            if self.training and dn_meta and 'output_known_lbs_bboxes' in dn_meta:
                single_pad, scalar = self.prep_for_dn(dn_meta)
                for i in range(len(targets_temp)):
                    if len(targets_temp[i]['labels']) > 0:
                        t = torch.range(0, len(targets_temp[i]['labels']) - 1).long().to(device)
                        t = t.unsqueeze(0).repeat(scalar, 1)
                        tgt_idx = t.flatten()
                        output_idx = (torch.tensor(range(scalar)) * single_pad).long().to(device).unsqueeze(1) + t
                        output_idx = output_idx.flatten()
                    else:
                        output_idx = tgt_idx = torch.tensor([]).long().to(device)
                    dn_pos_idx.append((output_idx, tgt_idx))
                    dn_neg_idx.append((output_idx + single_pad // 2, tgt_idx))
                # dn
                output_dn = {
                    "pred_logits": dn_meta["output_known_lbs_bboxes"]["pred_logits"],
                    "pred_boxes": dn_meta["output_known_lbs_bboxes"]["pred_boxes"]
                }

                l_dict = self.set_criterion(output_dn, targets_temp, task_info['losses'], num_boxes*scalar, dn_pos_idx)
                l_dict = {k + f'_dn': v for k, v in l_dict.items()}
                losses.update(l_dict)
                # dn aux_outputs
                for i in range(len(dn_meta["output_known_lbs_bboxes"]["aux_outputs"])):
                    output_aux_dn = {
                        "pred_logits": dn_meta['output_known_lbs_bboxes']["aux_outputs"][i]["pred_logits"],
                        "pred_boxes": dn_meta['output_known_lbs_bboxes']["aux_outputs"][i]["pred_boxes"],
                    }
                    l_dict = self.set_criterion(output_aux_dn, targets_temp, task_info['losses'], num_boxes*scalar, dn_pos_idx)
                    l_dict = {k + f'_dn_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)
            else:
                l_dict = dict()
                l_dict['loss_bbox_dn'] = torch.as_tensor(0.).to(device)
                l_dict['loss_giou_dn'] = torch.as_tensor(0.).to(device)
                l_dict['loss_ce_dn'] = torch.as_tensor(0.).to(device)
                losses.update(l_dict)
                for idx in range(self.num_decoder_layers-1):
                    l_dict = dict()
                    l_dict['loss_bbox_dn']=torch.as_tensor(0.).to(device)
                    l_dict['loss_giou_dn']=torch.as_tensor(0.).to(device)
                    l_dict['loss_ce_dn']=torch.as_tensor(0.).to(device)
                    l_dict = {k + f'_{idx}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses

    def get_num_boxes(self, targets, device):
        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(sum(t[key]['boxes'].shape[0] for key in t if isinstance(key, int)) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()
        return num_boxes
    
    def prep_for_dn(self,dn_meta):
        num_dn_groups,pad_size=dn_meta['num_dn_group'],dn_meta['pad_size']
        assert pad_size % num_dn_groups==0
        single_pad=pad_size//num_dn_groups

        return single_pad,num_dn_groups
