export PADDLE_TRAINERS_NUM=1
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"                                                                       

config=config/DPDETR_4scale_coco.py

coco_path=/bpfs/v2_mnt/VIS5/bpfsrw6/mscoco
coco_path=/home/ssd9/zhangbin33/coco_data/mscoco/

PYTHON=/home/kubernetes/dependency/pytorch_dsnet/bin/python
PYTHON=/home/ssd5/wangyunhao02/anaconda3/envs/pytorch_dsnet/bin/python

OMP_NUM_THREADS=1 $PYTHON -m torch.distributed.launch \
    --nproc_per_node=1 \
    main.py \
	-c $config \
    --coco_path $coco_path \
    --with_prompt_indicator \
    --with_objseq \
    --output_dir ./logs/ \
    --use_val_for_training \
    --with_two_stage_TI=80 \
    --options dn_scalar=100 \
    dn_label_coef=1.0 \
    dn_bbox_coef=1.0 \
#   --finetune_ignore label_enc.weight transformer.prompt_indicator.class_prompts \
#   --pretrain_model_path 