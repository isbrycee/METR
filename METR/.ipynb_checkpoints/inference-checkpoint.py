import os, sys
import torch, json
import numpy as np
import argparse

from main import build_model_main
from util.slconfig import SLConfig, DictAction
from datasets import build_dataset
from util.visualizer import COCOVisualizer
from util import box_ops

model_config_path = "config/METR_4scale_coco.py" # change the path of the model config file
model_checkpoint_path = "/home/ssd5/haojing/DP-DETR/checkpoint0049.pth" # change the path of the model checkpoint
# See our Model Zoo section in README.md for more details about our pretrained models.

def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--config_file', '-c', type=str, required=True)
    parser.add_argument('--options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file.')

    # dataset parameters
    parser.add_argument('--dataset_file', default='coco')
    parser.add_argument('--coco_path', type=str, default='/comp_robot/cv_public_dataset/COCO2017/')
    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')
    parser.add_argument('--fix_size', action='store_true')

    # training parameters
    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--note', default='',
                        help='add some notes to the experiment')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--pretrain_model_path', help='load from other checkpoint')
    parser.add_argument('--finetune_ignore', type=str, nargs='+')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--find_unused_params', action='store_true')
    

    parser.add_argument('--save_results', action='store_true')
    parser.add_argument('--save_log', action='store_true')

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--rank', default=0, type=int,
                        help='number of distributed processes')
    parser.add_argument("--local_rank", type=int, help='local rank for DistributedDataParallel')
    parser.add_argument('--amp', action='store_true',
                        help="Train with mixed precision")
    
    #
#     parser.add_argument('--num_class', default=80, type=int)
    parser.add_argument('--with_prompt_indicator', action='store_true')
    parser.add_argument('--with_objseq', action='store_true')
    parser.add_argument('--with_two_stage_TI', default=80, type=int, choices=[0, 20, 80])
    parser.add_argument('--eval_with_agnostic', action='store_true')
    parser.add_argument('--use_val_for_training', action='store_true')
    parser.add_argument('--use_plain_TI', action='store_true')
    parser.add_argument('--cls_asl_loss_weight', default=1, type=int)    
    parser.add_argument('--activation', default='relu')
    
    parser.add_argument('--large_scale_jitter', action='store_true')
    parser.add_argument('--color_jitter', action='store_true')
    parser.add_argument('--arrange_by_class', action='store_false')
    parser.add_argument('--min_keypoints_train', default=0, type=int)
    
    # 
#     parser.add_argument('--init_vectors', default='coco_clip_v2.npy', help='# .npy or .pth file, empty means random initialized')
    parser.add_argument('--fix_class_prompts', action='store_true')
    parser.add_argument('--prompt_indicator_num_blocks', default=2, type=int)
    parser.add_argument('--prompt_indicator_return_intermediate', action='store_false')
    parser.add_argument('--prompt_indicator_level_preserve', default=[])
    parser.add_argument('--prompt_indicator_no_self_attn', action='store_false')
    
    parser.add_argument('--classifier_type', default='dict')
    parser.add_argument('--classifier_hidden_dim', default=256, type=int)
    parser.add_argument('--classifier_num_layers', default=2)
    parser.add_argument('--classifier_init_prob', default=0.1)
    parser.add_argument('--classifier_num_points', default=1)
    parser.add_argument('--classifier_skip_and_init', action='store_true')
    parser.add_argument('--classifier_normalize_before', action='store_true')
    
    parser.add_argument('--retain_categories', action='store_false')
    parser.add_argument('--retention_policy_train_class_thr', default=0.0)
    parser.add_argument('--retention_policy_eval_class_thr', default=0.0)
    
    parser.add_argument('--prompt_indicator_losses', default=['asl'])
    parser.add_argument('--prompt_indicator_asl_optimized', action='store_false')
    parser.add_argument('--prompt_indicator_asl_gamma_pos', default=0.0)
    parser.add_argument('--prompt_indicator_asl_gamma_neg', default=2.0)
    parser.add_argument('--prompt_indicator_asl_clip', default=0.0)
    
    parser.add_argument('--set_class_normalization', default="none")
    parser.add_argument('--set_box_normalization', default="none")
    parser.add_argument('--class_normalization', default="num_box")
    parser.add_argument('--box_normalization', default="num_box")
    
    return parser


parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
args = parser.parse_args()
# load cfg file and update the args
print("Loading config file from {}".format(args.config_file))
cfg = SLConfig.fromfile(args.config_file)
if args.options is not None:
    cfg.merge_from_dict(args.options)

cfg_dict = cfg._cfg_dict.to_dict()
args_vars = vars(args)
for k,v in cfg_dict.items():
    if k not in args_vars:
        setattr(args, k, v)
    else:
        raise ValueError("Key {} can used by args only".format(k))
        

# args = SLConfig.fromfile(model_config_path) 
args.device = 'cuda' 
model, criterion, _, postprocessors = build_model_main(args)

checkpoint = torch.load(model_checkpoint_path, map_location='cpu')
model.load_state_dict(checkpoint['model'])
_ = model.eval()

# # load coco names
# with open('util/coco_id2name.json') as f:
#     id2name = json.load(f)
#     id2name = {int(k):v for k,v in id2name.items()}
    
id2name = {}
with open('/home/ssd5/haojing/class_merge/class_name_coco.txt') as f:
    lines = f.readlines()
    id = 0
    for line in lines:
        id2name[id] = line.strip()
        id += 1

args.dataset_file = 'coco'
args.coco_path = "/home/ssd9/zhangbin33/coco_data/mscoco/" # the path of coco
args.fix_size = False

dataset_val = build_dataset(image_set='val', args=args)

image, targets, target_for_class = dataset_val[0]
box_label = [id2name[int(item)] for item in targets['labels']]
gt_dict = {
    'boxes': targets['boxes'],
    'image_id': targets['image_id'],
    'size': targets['size'],
    'box_label': box_label,
}
vslzr = COCOVisualizer()
vslzr.visualize(image, gt_dict, savedir=None)

output = model.cuda()(image[None].cuda(), target_for_class = target_for_class)
print(output)
output = postprocessors['bbox'](output, torch.Tensor([[1.0, 1.0]]).cuda())[0]

thershold = 0.3 # set a thershold

scores = output['scores']
labels = output['labels']
boxes = box_ops.box_xyxy_to_cxcywh(output['boxes'])
select_mask = scores > thershold

box_label = [id2name[int(item)] for item in labels[select_mask]]
pred_dict = {
    'boxes': boxes[select_mask],
    'size': targets['size'],
    'box_label': box_label
}
vslzr.visualize(image, pred_dict, savedir=None)

