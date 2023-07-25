python tools/benchmark.py \
    --output_dir logs/test_flops \
    -c config/DPDETR_4scale_coco.py \
    --coco_path /home/ssd9/mscoco \
    --output_dir ./logs/ \
    --is_prompt_indicator \
    --is_embedding_align \
    --num_classes_for_CEM=80 \
    --options dn_scalar=100 \
    batch_size=1 \
    dn_label_coef=1.0 \
    dn_bbox_coef=1.0 \