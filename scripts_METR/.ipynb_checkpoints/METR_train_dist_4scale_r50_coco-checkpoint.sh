export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

config=config/METR_4scale_coco.py

coco_path=/home/ssd9/zhangbin33/coco_data/mscoco/

PYTHON=/home/ssd5/wangyunhao02/anaconda3/envs/pytorch_dsnet/bin/python

OMP_NUM_THREADS=1 $PYTHON -m torch.distributed.launch \
    --nproc_per_node=2 \
    main.py \
	-c $config \
    --coco_path $coco_path \
    --output_dir ./logs/ \
    --is_prompt_indicator \
    --is_embedding_align \
    --num_classes_for_CEM=80 \
    --options dn_scalar=100 \
    dn_label_coef=1.0 \
    dn_bbox_coef=1.0 \
#   --finetune_ignore label_enc.weight transformer.prompt_indicator.class_prompts \
#   --pretrain_model_path xxx.pth
