export PADDLE_TRAINERS_NUM=1
export CUDA_VISIBLE_DEVICES="0"                                                                       

config=config/DPDETR_4scale_coco.py
coco_path=/home/ssd9/mscoco/

OMP_NUM_THREADS=1 python inference.py \
	-c $config \
    --coco_path $coco_path \
    --output_dir ./logs/ \
    --is_prompt_indicator \
    --is_embedding_align \
    --num_classes_for_CEM=80 \
    --resume xxx.pth \
    --options dn_scalar=100 \
    dn_label_coef=1.0 \
    dn_bbox_coef=1.0 \