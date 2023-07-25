/home/ssd5/wangyunhao02/anaconda3/envs/pytorch_dsnet/bin/python tools/benchmark.py \
    --output_dir logs/test_flops \
    -c config/DINO/DINO_4scale.py \
    --options batch_size=1 \
    --coco_path /path/to/your/coco/dir