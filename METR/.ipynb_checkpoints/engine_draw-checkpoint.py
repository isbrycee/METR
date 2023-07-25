# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Train and eval functions used in main.py
"""

import math
import os
import sys
from typing import Iterable

from util.utils import slprint, to_device

import torch

import util.misc as utils
from datasets.coco_eval import CocoEvaluator
from datasets.panoptic_eval import PanopticEvaluator
import numpy as np


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
    print('######args.with_prompt_indicator########', args.with_prompt_indicator)
    print('######args.with_objseq########', args.with_objseq)
    print('######args.batch_size########', args.batch_size)
    print('######args.cls_loss_coef########', args.cls_loss_coef)
    print('######args.num_queries########', args.num_queries)
    print('######prompt_indicator_num_blocks######', args.prompt_indicator_num_blocks)
    print('######args.dec_layers######', args.dec_layers)
    print('######args.fix_class_prompts', args.fix_class_prompts)
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
#         for t in target_for_class:
#             for k, v in t.items():
#         target_for_class = [{k: v.to(device) for k, v in t.items()} for t in target_for_class]

        with torch.cuda.amp.autocast(enabled=args.amp):
            if need_tgt_for_training:
                outputs,prompt_indicator_loss_dict = model(samples, targets, target_for_class)
            else:
                outputs,prompt_indicator_loss_dict = model(samples, target_for_class = target_for_class)
        
            loss_dict = criterion(outputs, targets, target_for_class)
            if criterion_for_encoder is not None:
                loss_dict_encoder = criterion_for_encoder(outputs, targets, target_for_class)
                loss_dict.update(loss_dict_encoder)
            
            weight_dict = criterion.weight_dict
            if prompt_indicator_loss_dict:
                loss_dict.update(prompt_indicator_loss_dict)
                weight_dict.update({'cls_asl':0.25, 'cls_asl_0':0.25})
            # import ipdb; ipdb.set_trace()
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
    print('######args.with_prompt_indicator########', args.with_prompt_indicator)
    print('######args.with_objseq########', args.with_objseq)
    print('######args.batch_size########', args.batch_size)
    print('######args.cls_loss_coef########', args.cls_loss_coef)
    print('######args.num_queries########', args.num_queries)
    print('######prompt_indicator_num_blocks######', args.prompt_indicator_num_blocks)
    print('######args.dec_layers######', args.dec_layers)
    assert isinstance(args.use_dn, bool)
        
    coco_evaluator = CocoEvaluator(base_ds, iou_types, useCats=useCats, with_agnostic=args.eval_with_agnostic)
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
    
    import json
    import cv2
    with open('/home/ssd9/zhangbin33/coco_data/mscoco/annotations/instances_val2017.json', 'r') as f:
        json_data = json.load(f)
    id_filename = {}
    for i in json_data['images']:
        id_filename[i['id']] = i['file_name']
            
    for samples, targets, target_for_class in metric_logger.log_every(data_loader, 10, header, logger=logger):
        samples = samples.to(device)
        #import pdb; pdb.set_trace()
        #print('targets', targets)
        # targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        targets = [{k: to_device(v, device) for k, v in t.items()} for t in targets]

        with torch.cuda.amp.autocast(enabled=args.amp):
            if need_tgt_for_training:
                outputs, prompt_indicator_loss_dict = model(samples, targets, target_for_class)
            else:
                outputs,prompt_indicator_loss_dict = model(samples, target_for_class = target_for_class)
            # outputs = model(samples)

            loss_dict = criterion(outputs, targets,target_for_class)
#             if criterion_for_encoder is not None:
#                 loss_dict_encoder = criterion_for_encoder(outputs, targets, target_for_class)
#                 loss_dict.update(loss_dict_encoder)
        weight_dict = criterion.weight_dict
        
        if prompt_indicator_loss_dict:
            loss_dict.update(prompt_indicator_loss_dict)
            weight_dict.update({'cls_asl':0.25, 'cls_asl_0':0.25})
                
        #import pdb; pdb.set_trace()
        # loss_dict.keys()
        # dict_keys(['loss_ce', 'loss_bbox', 'loss_giou', 'loss_ce_0', 'loss_bbox_0', 'loss_giou_0', 'loss_ce_1', 'loss_bbox_1', 'loss_giou_1', 'loss_ce_2', 'loss_bbox_2', 'loss_giou_2', 'loss_ce_3', 'loss_bbox_3', 'loss_giou_3', 'loss_ce_4', 'loss_bbox_4', 'loss_giou_4', 'loss_ce_interm', 'loss_bbox_interm', 'loss_giou_interm', 'loss_xy_interm', 'loss_hw_interm', 'cardinality_error_interm', 'cls_asl', 'cls_asl_0', 'asl_class_error'])
        # weight_dict
        # {'loss_ce': 1.0, 'loss_bbox': 5.0, 'loss_giou': 2.0, 'loss_ce_dn': 1.0, 'loss_bbox_dn': 5.0, 'loss_giou_dn': 2.0, 'loss_ce_0': 1.0, 'loss_bbox_0': 5.0, 'loss_giou_0': 2.0, 'loss_ce_dn_0': 1.0, 'loss_bbox_dn_0': 5.0, 'loss_giou_dn_0': 2.0, 'loss_ce_1': 1.0, 'loss_bbox_1': 5.0, 'loss_giou_1': 2.0, 'loss_ce_dn_1': 1.0, 'loss_bbox_dn_1': 5.0, 'loss_giou_dn_1': 2.0, 'loss_ce_2': 1.0, 'loss_bbox_2': 5.0, 'loss_giou_2': 2.0, 'loss_ce_dn_2': 1.0, 'loss_bbox_dn_2': 5.0, 'loss_giou_dn_2': 2.0, 'loss_ce_3': 1.0, 'loss_bbox_3': 5.0, 'loss_giou_3': 2.0, 'loss_ce_dn_3': 1.0, 'loss_bbox_dn_3': 5.0, 'loss_giou_dn_3': 2.0, 'loss_ce_4': 1.0, 'loss_bbox_4': 5.0, 'loss_giou_4': 2.0, 'loss_ce_dn_4': 1.0, 'loss_bbox_dn_4': 5.0, 'loss_giou_dn_4': 2.0, 'loss_ce_interm': 1.0, 'loss_bbox_interm': 5.0, 'loss_giou_interm': 2.0, 'cls_asl': 1, 'cls_asl_0': 1}

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
        # [scores: [100], labels: [100], boxes: [100, 4]] x B
        if 'segm' in postprocessors.keys():
            target_sizes = torch.stack([t["size"] for t in targets], dim=0)
            results = postprocessors['segm'](results, outputs, orig_target_sizes, target_sizes)
        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        # import ipdb; ipdb.set_trace()
        
        mapped_labels_ = {0: 0, 1: 28}
        
        mapped_labels = {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7, 7: 8, 8: 9, 9: 10, 10: 11, 11: 13, 12: 14, 13: 15, 14: 16, 15: 17, 16: 18, 17: 19, 18: 20, 19: 21, 20: 22, 21: 23, 22: 24, 23: 25, 24: 27, 25: 28, 26: 31, 27: 32, 28: 33, 29: 34, 30: 35, 31: 36, 32: 37, 33: 38, 34: 39, 35: 40, 36: 41, 37: 42, 38: 43, 39: 44, 40: 46, 41: 47, 42: 48, 43: 49, 44: 50, 45: 51, 46: 52, 47: 53, 48: 54, 49: 55, 50: 56, 51: 57, 52: 58, 53: 59, 54: 60, 55: 61, 56: 62, 57: 63, 58: 64, 59: 65, 60: 67, 61: 70, 62: 72, 63: 73, 64: 74, 65: 75, 66: 76, 67: 77, 68: 78, 69: 79, 70: 80, 71: 81, 72: 82, 73: 84, 74: 85, 75: 86, 76: 87, 77: 88, 78: 89, 79: 90}
        
            
        id_name = {}
        for i in json_data['categories']:
            id_name[i['id']] = i['name']
        print(id_name)
        # {1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane', 6: 'bus', 7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light', 11: 'fire hydrant', 13: 'stop sign', 14: 'parking meter', 15: 'bench', 16: 'bird', 17: 'cat', 18: 'dog', 19: 'horse', 20: 'sheep', 21: 'cow', 22: 'elephant', 23: 'bear', 24: 'zebra', 25: 'giraffe', 27: 'backpack', 28: 'umbrella', 31: 'handbag', 32: 'tie', 33: 'suitcase', 34: 'frisbee', 35: 'skis', 36: 'snowboard', 37: 'sports ball', 38: 'kite', 39: 'baseball bat', 40: 'baseball glove', 41: 'skateboard', 42: 'surfboard', 43: 'tennis racket', 44: 'bottle', 46: 'wine glass', 47: 'cup', 48: 'fork', 49: 'knife', 50: 'spoon', 51: 'bowl', 52: 'banana', 53: 'apple', 54: 'sandwich', 55: 'orange', 56: 'broccoli', 57: 'carrot', 58: 'hot dog', 59: 'pizza', 60: 'donut', 61: 'cake', 62: 'chair', 63: 'couch', 64: 'potted plant', 65: 'bed', 67: 'dining table', 70: 'toilet', 72: 'tv', 73: 'laptop', 74: 'mouse', 75: 'remote', 76: 'keyboard', 77: 'cell phone', 78: 'microwave', 79: 'oven', 80: 'toaster', 81: 'sink', 82: 'refrigerator', 84: 'book', 85: 'clock', 86: 'vase', 87: 'scissors', 88: 'teddy bear', 89: 'hair drier', 90: 'toothbrush'}
            
        selected_names = ['000000034139.jpg', '000000025057.jpg']
            
        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        img_name = id_filename[targets[0]['image_id'].item()].split('/')[-1]

        img = cv2.imread('/home/ssd9/zhangbin33/coco_data/mscoco/val2017/'+img_name)
        score = res[targets[0]['image_id'].item()]['scores']
        topk = np.sum(score.cpu().numpy()>0.2)
        if topk>0:
            print(img_name, 'topk', topk)
            for num,i in enumerate(res[targets[0]['image_id'].item()]['boxes'][:topk]):
                x, y, x2, y2 = i[0].item(), i[1].item(), i[2].item(), i[3].item()
                cv2.rectangle(img, (int(x), int(y)), (int(x2), int(y2)), (0, 255, 0), 2)

                label = res[targets[0]['image_id'].item()]['labels'][:topk][num].item()
                print(label)
                cv2.putText(img, id_name[mapped_labels[mapped_labels_[label]]], (int(x), int(y)), cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,255),1)
                #cv2.putText(img, id_name[mapped_labels[label]], (int(x), int(y)), cv2.FONT_HERSHEY_COMPLEX,0.7,(0,0,255),1)
            #cv2.imwrite('./val_visual_0.2/'+img_name, img)        
            cv2.imwrite(img_name, img)        
        
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
            # res_score = outputs['res_score']
            # res_label = outputs['res_label']
            # res_bbox = outputs['res_bbox']
            # res_idx = outputs['res_idx']
            # import ipdb; ipdb.set_trace()

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
                # import ipdb;ipdb.set_trace()

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
        
        # output_state_dict['gt_info'] = torch.cat(output_state_dict['gt_info'])
        # output_state_dict['res_info'] = torch.cat(output_state_dict['res_info'])
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

    # import ipdb; ipdb.set_trace()

    return stats, coco_evaluator


@torch.no_grad()
def test(model, criterion, postprocessors, data_loader, base_ds, device, output_dir, wo_class_error=False, args=None, logger=None):
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    # if not wo_class_error:
    #     metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    # coco_evaluator = CocoEvaluator(base_ds, iou_types)
    # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]

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
        # import ipdb; ipdb.set_trace()
        # targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        targets = [{k: to_device(v, device) for k, v in t.items()} for t in targets]

        outputs = model(samples)
        # loss_dict = criterion(outputs, targets)
        # weight_dict = criterion.weight_dict

        # # reduce losses over all GPUs for logging purposes
        # loss_dict_reduced = utils.reduce_dict(loss_dict)
        # loss_dict_reduced_scaled = {k: v * weight_dict[k]
        #                             for k, v in loss_dict_reduced.items() if k in weight_dict}
        # loss_dict_reduced_unscaled = {f'{k}_unscaled': v
        #                               for k, v in loss_dict_reduced.items()}
        # metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
        #                      **loss_dict_reduced_scaled,
        #                      **loss_dict_reduced_unscaled)
        # if 'class_error' in loss_dict_reduced:
        #     metric_logger.update(class_error=loss_dict_reduced['class_error'])

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors['bbox'](outputs, orig_target_sizes, not_to_xyxy=True)
        # [scores: [100], labels: [100], boxes: [100, 4]] x B
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
        
    # gather the stats from all processes
    # metric_logger.synchronize_between_processes()
    # print("Averaged stats:", metric_logger)
    # if coco_evaluator is not None:
    #     coco_evaluator.synchronize_between_processes()
    # if panoptic_evaluator is not None:
    #     panoptic_evaluator.synchronize_between_processes()

    # # accumulate predictions from all images
    # if coco_evaluator is not None:
    #     coco_evaluator.accumulate()
    #     coco_evaluator.summarize()
        
    # panoptic_res = None
    # if panoptic_evaluator is not None:
    #     panoptic_res = panoptic_evaluator.summarize()
    # stats = {k: meter.global_avg for k, meter in metric_logger.meters.items() if meter.count > 0}
    # if coco_evaluator is not None:
    #     if 'bbox' in postprocessors.keys():
    #         stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
    #     if 'segm' in postprocessors.keys():
    #         stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()
    # if panoptic_res is not None:
    #     stats['PQ_all'] = panoptic_res["All"]
    #     stats['PQ_th'] = panoptic_res["Things"]
    #     stats['PQ_st'] = panoptic_res["Stuff"]

    # import ipdb; ipdb.set_trace()

    return final_res
