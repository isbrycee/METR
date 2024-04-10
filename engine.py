# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Train and eval functions used in main.py
"""
from fileinput import filename
import json
import math
import os
import sys
from traceback import print_tb
from typing import Iterable
import numpy as np
from util import box_ops
from torchvision.ops import nms
from util.utils import slprint, to_device
import torch
import util.misc as utils
from datasets.coco_eval import CocoEvaluator
from datasets.panoptic_eval import PanopticEvaluator
from torch import Tensor


def jitter_boxes(boxes, jitter_range=0.005):
    jitter = jitter_range * (boxes[:, 2:].max(dim=1).values[0])

    dx = torch.randn_like(boxes[:, :1]) * jitter
    dy = torch.randn_like(boxes[:, 1:2]) * jitter
    dw = torch.randn_like(boxes[:, 2:3]) * jitter
    dh = torch.randn_like(boxes[:, 3:]) * jitter

    # 对目标框进行抖动
    jittered_boxes = torch.cat((boxes[:, :1] + dx, boxes[:, 1:2] + dy, boxes[:, 2:3] + dw, boxes[:, 3:] + dh), dim=1)

    return jittered_boxes

def box_area(boxes: Tensor) -> Tensor:
    """
    Computes the area of a set of bounding boxes, which are specified by its
    (x1, y1, x2, y2) coordinates.
    Arguments:
        boxes (Tensor[N, 4]): boxes for which the area will be computed. They
            are expected to be in (x1, y1, x2, y2) format
    Returns:
        area (Tensor[N]): area for each box
    """
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
 
 
def box_iou(boxes1: Tensor, boxes2: Tensor) -> Tensor:
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        boxes1 (Tensor[N, 4])
        boxes2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise IoU values for every element in boxes1 and boxes2
    """
    area1 = box_area(boxes1)  # 每个框的面积 (N,)
    area2 = box_area(boxes2)  # (M,)
 
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2] # N中一个和M个比较； 所以由N，M 个
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]
 
    wh = (rb - lt).clamp(min=0)  # [N,M,2]  #小于0的为0  clamp 钳；夹钳；
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]  
 
    iou = inter / (area1[:, None] + area2 - inter)
    return iou  # NxM， boxes1中每个框和boxes2中每个框的IoU值；
 
 
def nms(boxes: Tensor, scores: Tensor, iou_threshold: float):
    """
    :param boxes: [N, 4]， 此处传进来的框，是经过筛选（NMS之前选取过得分TopK）之后， 在传入之前处理好的；
    :param scores: [N]
    :param iou_threshold: 0.7
    :return:
    """
    keep = []  # 最终保留的结果， 在boxes中对应的索引；
    valid_keep = []
    idxs = scores.argsort()  # 值从小到大的 索引
    while idxs.numel() > 0:  # 循环直到null； numel()： 数组元素个数
        # 得分最大框对应的索引, 以及对应的坐标
        max_score_index = idxs[-1]
        max_score_box = boxes[max_score_index][None, :]  # [1, 4]
        keep.append(max_score_index)
        if idxs.size(0) == 1:  # 就剩余一个框了；
            break
        idxs = idxs[:-1]  # 将得分最大框 从索引中删除； 剩余索引对应的框 和 得分最大框 计算IoU；
        other_boxes = boxes[idxs]  # [?, 4]
        ious = box_iou(max_score_box, other_boxes)  # 一个框和其余框比较 1XM
        idxs = idxs[ious[0] <= iou_threshold]
 
    condition = ((boxes[:, 0] > boxes[:, 2]) | (boxes[:, 1] > boxes[:, 3]))
    indices = torch.nonzero(condition).squeeze()
    
    for ind in keep:
        if ind.item() not in indices:
            valid_keep.append(ind)
    keep = idxs.new(valid_keep)  # Tensor

    saved_boxes = boxes[keep]
    return keep, saved_boxes

def generate_psudou_bbox(outputs, targers, class_index, score_thresh=0.90, iou_threshold=0.90):
    src_logits = outputs['pred_logits']
    device = src_logits.device
    src_score = src_logits.sigmoid()
    src_saved_idx = (src_score > score_thresh).nonzero()
    if len(src_saved_idx) == 0:
        return targers
    src_saved_class_id = class_index[src_saved_idx[:, 0]]
    
    src_saved_scores = src_score[src_saved_idx[:, 0], src_saved_idx[:, 1]]
    src_boxes = outputs['pred_boxes']
    src_saved_boxes = src_boxes[src_saved_idx[:, 0], src_saved_idx[:, 1], :]
    
    src_saved_boxes_xyxy = box_ops.box_cxcywh_to_xyxy(src_saved_boxes)
    img_w, img_h = targers[0]['size'][0].item(), targers[0]['size'][1].item()
    scale_fct = torch.Tensor([img_w, img_h, img_w, img_h]).to(device)
    src_saved_boxes_xyxy_ori = src_saved_boxes_xyxy * scale_fct
    
    # nms
    keep, kept_bboxes_xyxy = nms(src_saved_boxes_xyxy_ori, src_saved_scores, iou_threshold)
    kept_bboxes = src_saved_boxes[keep]
    kept_bboxes = jitter_boxes(kept_bboxes)
    kept_bboxes_xyxy = box_ops.box_cxcywh_to_xyxy(kept_bboxes)

    src_saved_idx = src_saved_idx[keep]
    src_src_saved_boxes_wh = targers[0]['size'] * kept_bboxes_xyxy[:, 2:]
    src_src_saved_boxes_areas = torch.prod(src_src_saved_boxes_wh, dim=-1)
    kept_label = src_saved_class_id[keep]

    bs = len(targers)
    assert bs == 1
    for i in range(bs):
        boxes = targers[i]['boxes']
        combined_boxes = torch.cat((boxes, kept_bboxes), dim=0)
        targers[i]['boxes'] = combined_boxes.detach()

        labels = targers[i]['labels']
        combined_labels = torch.cat((labels, kept_label), dim=0)
        targers[i]['labels'] = combined_labels.detach()

        area = targers[i]['area']
        combined_area = torch.cat((area, src_src_saved_boxes_areas), dim=0)
        targers[i]['area'] = combined_area.detach()

        iscrowd = targers[i]['iscrowd']
        combined_iscrowd = torch.zeros_like(combined_labels)
        targers[i]['iscrowd'] = combined_iscrowd.detach()

        multi_label_onehot = targers[i]['multi_label_onehot']
        combined_multi_label_onehot = multi_label_onehot.scatter_(0, kept_label, 1)
        targers[i]['multi_label_onehot'] = combined_multi_label_onehot.detach()

        class_label = targers[i]['class_label']
        combined_class_label = torch.unique(torch.cat((class_label, kept_label)))
        targers[i]['class_label'] = combined_class_label.detach()

    return targers

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module, criterion_for_encoder: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0, 
                    wo_class_error=False, lr_scheduler=None, args=None, logger=None, ema_m=None):
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    try:
        need_tgt_for_training = args.use_dn
    except:
        need_tgt_for_training = False
    
    print('######args.two_stage_type########', args.two_stage_type)
    print('######args.use_dn########', args.use_dn)
    print('######args.is_prompt_indicator########', args.is_prompt_indicator)
    print('######args.is_embedding_align########', args.is_embedding_align)
    print('######args.batch_size########', args.batch_size)
    print('######args.cls_loss_coef########', args.cls_loss_coef)
    print('######args.num_queries########', args.num_queries)
    print('######args.num_select########', args.num_select)
    print('######args.fix_class_prompts########', args.fix_class_prompts)
    print('######prompt_indicator_num_blocks######', args.prompt_indicator_num_blocks)
    print('######args.dec_layers######', args.dec_layers)
    print('######args.param_dict_type########', args.param_dict_type)
    print('######args.text_embed_type########', args.text_embed_type)
    print('######args.eval_map_type########', args.eval_map_type)
    print('######args.use_plain_CEM########', args.use_plain_CEM)
    print('######args.cls_asl_loss_weight########', args.cls_asl_loss_weight)
    print('######args.dataset_type########', args.dataset_type)
    assert isinstance(args.use_dn, bool)
    
    model.train()
    criterion.train()
    if criterion_for_encoder is not None:
        criterion_for_encoder.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    if not wo_class_error:
        metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    _cnt = 0
    for samples, targets, target_for_class in metric_logger.log_every(data_loader, print_freq, header, logger=logger):

        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        with torch.cuda.amp.autocast(enabled=args.amp):
            if need_tgt_for_training:
                outputs,prompt_indicator_loss_dict = model(samples, targets, target_for_class)
            else:
                outputs,prompt_indicator_loss_dict = model(samples, target_for_class = target_for_class)
            
            if args.pseudo_boxes and epoch > 6:
                class_index = outputs['class_index']
                targets = generate_psudou_bbox(outputs, targets, class_index, args.thresh_pseudo_boxes)

            loss_dict = criterion(outputs, targets, target_for_class)
            
            if criterion_for_encoder is not None:
                loss_dict_encoder = criterion_for_encoder(outputs, targets, target_for_class)
                loss_dict.update(loss_dict_encoder)
            
            weight_dict = criterion.weight_dict
            if prompt_indicator_loss_dict:
                loss_dict.update(prompt_indicator_loss_dict)
                weight_dict.update({'cls_asl':args.cls_asl_loss_weight, 'cls_asl_0':args.cls_asl_loss_weight})
                weight_dict.update({'cls_bce':args.cls_asl_loss_weight, 'cls_bec_0':args.cls_asl_loss_weight})
            losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())
        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        # amp backward function
        if args.amp: #
            optimizer.zero_grad()
            scaler.scale(losses).backward()
            if max_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            scaler.step(optimizer)
            scaler.update()
        else: # here
            # original backward function
            optimizer.zero_grad()
            losses.backward()
            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()

        if args.onecyclelr:
            lr_scheduler.step()
        if args.use_ema:
            if epoch >= args.ema_epoch:
                ema_m.update(model)

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)

        if 'class_error' in loss_dict_reduced:
            metric_logger.update(class_error=loss_dict_reduced['class_error'])
        if 'class_error_interm' in loss_dict_reduced:
            metric_logger.update(interm_class_error=loss_dict_reduced['class_error_interm'])
        if 'asl_class_error' in loss_dict_reduced:
            metric_logger.update(asl_class_error=loss_dict_reduced['asl_class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        _cnt += 1
        if args.debug:
            if _cnt % 15 == 0:
                print("BREAK!"*5)
                break

    if getattr(criterion, 'loss_weight_decay', False):
        criterion.loss_weight_decay(epoch=epoch)
    if getattr(criterion, 'tuning_matching', False):
        criterion.tuning_matching(epoch)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    resstat = {k: meter.global_avg for k, meter in metric_logger.meters.items() if meter.count > 0}
    if getattr(criterion, 'loss_weight_decay', False):
        resstat.update({f'weight_{k}': v for k,v in criterion.weight_dict.items()})
    return resstat


@torch.no_grad()
def evaluate(model, criterion, criterion_for_encoder, postprocessors, data_loader, base_ds, device, output_dir, wo_class_error=False, args=None, logger=None):
    try:
        need_tgt_for_training = args.use_dn
    except:
        need_tgt_for_training = False

    model.eval()
    criterion.eval()
    if criterion_for_encoder is not None:
        criterion_for_encoder.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    if not wo_class_error:
        metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    useCats = True
    try:
        useCats = args.useCats
    except:
        useCats = True

    if not useCats:
        print("useCats: {} !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!".format(useCats))
        
    print('######args.two_stage_type########', args.two_stage_type)
    print('######args.use_dn########', args.use_dn)
    print('######args.is_prompt_indicator########', args.is_prompt_indicator)
    print('######args.is_embedding_align########', args.is_embedding_align)
    print('######args.batch_size########', args.batch_size)
    print('######args.cls_loss_coef########', args.cls_loss_coef)
    print('######args.num_queries########', args.num_queries)
    print('######args.num_select########', args.num_select)
    print('######args.fix_class_prompts########', args.fix_class_prompts)
    print('######prompt_indicator_num_blocks######', args.prompt_indicator_num_blocks)
    print('######args.dec_layers######', args.dec_layers)
    print('######args.param_dict_type########', args.param_dict_type)
    print('######args.text_embed_type########', args.text_embed_type)
    print('######args.eval_map_type########', args.eval_map_type)
    print('######args.use_plain_CEM########', args.use_plain_CEM)
    print('######args.cls_asl_loss_weight########', args.cls_asl_loss_weight)
    print('######args.dataset_type########', args.dataset_type)
    assert isinstance(args.use_dn, bool)
        
    coco_evaluator = CocoEvaluator(base_ds, iou_types, useCats=useCats, with_agnostic=args.eval_with_agnostic,eval_map_type=args.eval_map_type)
    # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]

    panoptic_evaluator = None
    if 'panoptic' in postprocessors.keys():
        panoptic_evaluator = PanopticEvaluator(
            data_loader.dataset.ann_file,
            data_loader.dataset.ann_folder,
            output_dir=os.path.join(output_dir, "panoptic_eval"),
        )

    _cnt = 0
    output_state_dict = {} # for debug only

    for samples, targets, target_for_class in metric_logger.log_every(data_loader, 10, header, logger=logger):
        samples = samples.to(device)
        targets = [{k: to_device(v, device) for k, v in t.items()} for t in targets]

        with torch.cuda.amp.autocast(enabled=args.amp):
            if need_tgt_for_training:
                outputs, prompt_indicator_loss_dict = model(samples, targets, target_for_class)
            else:
                outputs,prompt_indicator_loss_dict = model(samples, target_for_class = target_for_class)

            loss_dict = criterion(outputs, targets,target_for_class)
            if criterion_for_encoder is not None:
                loss_dict_encoder = criterion_for_encoder(outputs, targets, target_for_class)
                loss_dict.update(loss_dict_encoder)
        weight_dict = criterion.weight_dict
        
        if prompt_indicator_loss_dict:
            loss_dict.update(prompt_indicator_loss_dict)
            weight_dict.update({'cls_asl':0.25, 'cls_asl_0':0.25})
            
        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
                             **loss_dict_reduced_scaled,
                             **loss_dict_reduced_unscaled)
        if 'class_error' in loss_dict_reduced:
            metric_logger.update(class_error=loss_dict_reduced['class_error'])
        if 'class_error_interm' in loss_dict_reduced:
            metric_logger.update(interm_class_error=loss_dict_reduced['class_error_interm'])
        if 'asl_class_error' in loss_dict_reduced:
            metric_logger.update(asl_class_error=loss_dict_reduced['asl_class_error'])
            
        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors['bbox'](outputs, orig_target_sizes)
        if 'segm' in postprocessors.keys():
            target_sizes = torch.stack([t["size"] for t in targets], dim=0)
            results = postprocessors['segm'](results, outputs, orig_target_sizes, target_sizes)
        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        
        if coco_evaluator is not None:
            coco_evaluator.update(res)

        if panoptic_evaluator is not None:
            res_pano = postprocessors["panoptic"](outputs, target_sizes, orig_target_sizes)
            for i, target in enumerate(targets):
                image_id = target["image_id"].item()
                file_name = f"{image_id:012d}.png"
                res_pano[i]["image_id"] = image_id
                res_pano[i]["file_name"] = file_name

            panoptic_evaluator.update(res_pano)
        
        if args.save_results:
            for i, (tgt, res, outbbox) in enumerate(zip(targets, results, outputs['pred_boxes'])):
                """
                pred vars:
                    K: number of bbox pred
                    score: Tensor(K),
                    label: list(len: K),
                    bbox: Tensor(K, 4)
                    idx: list(len: K)
                tgt: dict.

                """
                # compare gt and res (after postprocess)
                gt_bbox = tgt['boxes']
                gt_label = tgt['labels']
                gt_info = torch.cat((gt_bbox, gt_label.unsqueeze(-1)), 1)
                
                # img_h, img_w = tgt['orig_size'].unbind()
                # scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=0)
                # _res_bbox = res['boxes'] / scale_fct
                _res_bbox = outbbox
                _res_prob = res['scores']
                _res_label = res['labels']
                res_info = torch.cat((_res_bbox, _res_prob.unsqueeze(-1), _res_label.unsqueeze(-1)), 1)

                if 'gt_info' not in output_state_dict:
                    output_state_dict['gt_info'] = []
                output_state_dict['gt_info'].append(gt_info.cpu())

                if 'res_info' not in output_state_dict:
                    output_state_dict['res_info'] = []
                output_state_dict['res_info'].append(res_info.cpu())

            # # for debug only
            # import random
            # if random.random() > 0.7:
            #     print("Now let's break")
            #     break

        _cnt += 1
        if args.debug:
            if _cnt % 15 == 0:
                print("BREAK!"*5)
                break

    if args.save_results:
        import os.path as osp
        
        savepath = osp.join(args.output_dir, 'results-{}.pkl'.format(utils.get_rank()))
        print("Saving res to {}".format(savepath))
        torch.save(output_state_dict, savepath)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()
    if panoptic_evaluator is not None:
        panoptic_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
        
    panoptic_res = None
    if panoptic_evaluator is not None:
        panoptic_res = panoptic_evaluator.summarize()
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items() if meter.count > 0}
    if coco_evaluator is not None:
        if 'bbox' in postprocessors.keys():
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
        if 'segm' in postprocessors.keys():
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()
    if panoptic_res is not None:
        stats['PQ_all'] = panoptic_res["All"]
        stats['PQ_th'] = panoptic_res["Things"]
        stats['PQ_st'] = panoptic_res["Stuff"]

    return stats, coco_evaluator


@torch.no_grad()
def test(model, criterion, postprocessors, data_loader, base_ds, device, output_dir, wo_class_error=False, args=None, logger=None):
    model.eval()
    criterion.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())

    panoptic_evaluator = None
    if 'panoptic' in postprocessors.keys():
        panoptic_evaluator = PanopticEvaluator(
            data_loader.dataset.ann_file,
            data_loader.dataset.ann_folder,
            output_dir=os.path.join(output_dir, "panoptic_eval"),
        )

    final_res = []
    for samples, targets in metric_logger.log_every(data_loader, 10, header, logger=logger):
        samples = samples.to(device)
        targets = [{k: to_device(v, device) for k, v in t.items()} for t in targets]
        outputs = model(samples)
        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors['bbox'](outputs, orig_target_sizes, not_to_xyxy=True)
        if 'segm' in postprocessors.keys():
            target_sizes = torch.stack([t["size"] for t in targets], dim=0)
            results = postprocessors['segm'](results, outputs, orig_target_sizes, target_sizes)
        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        for image_id, outputs in res.items():
            _scores = outputs['scores'].tolist()
            _labels = outputs['labels'].tolist()
            _boxes = outputs['boxes'].tolist()
            for s, l, b in zip(_scores, _labels, _boxes):
                assert isinstance(l, int)
                itemdict = {
                        "image_id": int(image_id), 
                        "category_id": l, 
                        "bbox": b, 
                        "score": s,
                        }
                final_res.append(itemdict)

    if args.output_dir:
        import json
        with open(args.output_dir + f'/results{args.rank}.json', 'w') as f:
            json.dump(final_res, f)        

    return final_res
