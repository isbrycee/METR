import os
import sys
import json
import numpy as np


def cal_iou(gtx1, gty1, gtx2, gty2, prex1, prey1, prex2, prey2):
    w = min(prex2, gtx2) - max(gtx1, prex1)
    h = min(gty2, prey2) - max(gty1, prey1)
    if w <= 0 or h <= 0:
        # print('w or h <= 0')
        return 0
    sa = (gtx2 - gtx1) * (gty2 - gty1)
    sb = (prex2 - prex1) * (prey2 - prey1)
    cross = float(w) * float(h)
    return cross / (sa + sb - cross)

def gtresult(gt_data):
    gtdict = {}
    for anno in gt_data['annotations']:
        id = anno["image_id"]
        if id not in gtdict:
            gtdict[id] = [anno]
        else:
            gtdict[id].append(anno)
    print(len(gtdict))
    return gtdict

def predresult(pred_data):
    preddict = {}
    for res in pred_data:
        id = res['image_id']
        if id not in preddict:
            preddict[id] = [res]
        else:
            preddict[id].append(res)
    print(len(preddict))
    return preddict

def eval_one_image(gtres, predres, thr, iou):
    preindex = 0
    evalindex = 0 #false
    for pred in predres:
        score = pred['score']
        if score < thr: continue
        preindex = 1
        predbbox = pred['bbox']
        prex1, prey1, prex2, prey2 = predbbox[0], predbbox[1], predbbox[0] + predbbox[2], predbbox[1] + predbbox[3]
        catid = pred['category_id']
        for gt in gtres:
            gt_catid = gt['category_id']
            if catid != gt_catid: continue
            gtbbox = gt['bbox']
            gtx1, gty1, gtx2, gty2 = gtbbox[0], gtbbox[1], gtbbox[0] + gtbbox[2], gtbbox[1] + gtbbox[3]
            if cal_iou(gtx1, gty1, gtx2, gty2, prex1, prey1, prex2, prey2) < iou: 
                continue
            else:
                evalindex = 1 #true
                return evalindex, preindex
    return evalindex, preindex

def primages(gtdict, preddict, thr, iou):
    right = 0
    gt_num = len(gtdict)
    pred_num = 0
    for id in gtdict:
        evalindex, predindex = eval_one_image(gtdict[id], preddict[id], thr, iou)
        right += evalindex
        pred_num += predindex
    precision = float(right) / pred_num
    recall = float(right) / gt_num
    print("score_thr:", round(thr, 1), "right:", right, "gt_num:", gt_num, "pred_num:", pred_num, \
          "recall:", round(recall * 100, 2), "pre:", round(precision * 100, 2),)
    return round(precision * 100, 2), round(recall * 100, 2)
def pr(cls_chen, score_thr, gt_data, pred_data):
    right = 0
    gt_num = 0

    have_matched_gt_id = set()
    have_matched_pred_id = set()
    for gt in gt_data['annotations']:
        catid = gt['category_id']
        bbox = gt['bbox']
        if catid == cls_chen:
            gt_num += 1

            x1, y1 = bbox[0], bbox[1]
            x2, y2 = x1 + bbox[2], y1 + bbox[3]
            max_iou_info = {'iou': 0, 'cls': 'none', 'matched_gt_id': -1, 'matched_pred_id': -1}
            for pred in pred_data:
                if pred['image_id'] == gt['image_id'] and pred['score'] > score_thr and pred['category_id'] == cls_chen:
                    px1, py1, px2, py2 = pred['bbox']  # eval results: x,y,w,h
                    px2, py2 = px1 + px2, py1 + py2
                    iou = cal_iou(x1, y1, x2, y2, px1, py1, px2, py2)
                    if iou > max_iou_info['iou'] and pred['category_id'] == catid:
                        max_iou_info['iou'] = iou
                        max_iou_info['cls'] = pred['category_id']
                        max_iou_info['matched_gt_id'] = gt['id']
                        max_iou_info['matched_pred_id'] = pred_data.index(pred)
            if max_iou_info['iou'] > iou_thr_for_positive and max_iou_info['cls'] == catid and max_iou_info['matched_gt_id'] != -1:
                if max_iou_info['matched_pred_id'] not in have_matched_pred_id:
                    right += 1
                    have_matched_gt_id.add(max_iou_info['matched_gt_id'])
                    have_matched_pred_id.add(max_iou_info['matched_pred_id'])

    pred_num = 0
    for pred in pred_data:
        if pred['score'] > score_thr and pred['category_id'] == cls_chen:
            pred_num += 1

    if gt_num != 0:
        recall = float(right) / gt_num
    else:
        recall = 0
    if pred_num != 0:
        precision = float(right) / pred_num
    else:
        precision = 0
    #import pdb;pdb.set_trace()
    if (recall + precision) != 0:
        f1 = 2 * recall * precision / (recall + precision)
    else:
        f1 = 0
    print("score:", round(score_thr, 1), "right:", right, "gt_num:", gt_num, "pred_num:", pred_num)
    print(classs_name[cls_chen], "re:", round(recall * 100, 2), "pre:", round(precision * 100, 2), "f1:",
          round(f1 * 100, 2))
    print(classs_name[cls_chen], "re:", round(recall * 100, 2), "pre:", round(precision * 100, 2))
    return pred_num


def write_gt_to_pred(gt_data):
    gt_data_pred = []
    for i in gt_data['annotations']:
        gt_data_pred.append({'image_id': i['image_id'], 'bbox': i['bbox'], 'category_id': i['category_id'], 'score': 1})
    return gt_data_pred


if __name__ == '__main__':
    gt_json = 'all-no-few-val.json'
    eval_json = '/home/disk1/vis/zhangbin_data/data_test_TCL/testdata/55UNB/bbox.json'
    eval_json = 'bbox.json'
    iou_thr_for_positive = 0.1

    with open(gt_json, 'r') as f:
        gt_data = json.load(f)
    classs_name = {i['id']: i['name'] for i in gt_data['categories']}
    
    print('classs_name ', classs_name)
    with open(eval_json, 'r') as f:
        pred_data = json.load(f)
    
    gtdict =  gtresult(gt_data)
    preddict = predresult(pred_data)
    gt_data = write_gt_to_pred(gt_data)
    preddict = gtresult(gt_data)
    all_score_thr = (0.1,0.2,0.3,0.4,0.5)#list(np.arange(0.0, 1.0, 0.1))
    all_pred_num = 0
    """
    for cls_chen in list(classs_name.keys()):
        for i, score_thr in enumerate(all_score_thr):
            pred_num = pr(cls_chen, score_thr, gt_data, pred_data)
            if i == 0:
                all_pred_num += pred_num
    assert (all_pred_num == len(pred_data))
    """
    for i, score_thr in enumerate(all_score_thr):
        print (score_thr)
        pre, recall = primages(gtdict, preddict, score_thr, iou_thr_for_positive)

    #print('#########The iou threshold for positive: ############', iou_thr_for_positive)