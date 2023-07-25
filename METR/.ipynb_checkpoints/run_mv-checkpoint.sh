source ~/.bpfs_env
mv resnet50-19c8e357.pth resnet50-0676ba61.pth
mkdir -p /root/.cache/torch/hub/checkpoints/
mv resnet50-0676ba61.pth /root/.cache/torch/hub/checkpoints/

