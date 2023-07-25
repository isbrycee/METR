export PADDLE_TRAINERS_NUM=4
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"                                                                       
ADDR=10.174.136.86
NNODES=4
RANK=0
PORT=44148

config=config/DPDETR_4scale_ImageNetbox.py

coco_path=/bpfs/v2_mnt/VIS5/bpfsrw6/ImageNetBox_2014

PYTHON=/home/kubernetes/dependency/pytorch_dsnet/bin/python

OMP_NUM_THREADS=1 $PYTHON -m torch.distributed.launch \
    --nnodes=$NNODES \
    --node_rank=$RANK \
    --master_addr=$ADDR \
    --master_port=$PORT \
    --nproc_per_node=8 \
    main.py \
	-c $config \
    --coco_path $coco_path \
    --with_prompt_indicator \
    --with_objseq \
    --output_dir ./logs/ \
    --with_two_stage_TI=80 \
	--options dn_scalar=100 \
    dn_label_coef=1.0 \
    dn_bbox_coef=1.0 \
#   --finetune_ignore label_enc.weight transformer.prompt_indicator.class_prompts \
#   --pretrain_model_path 