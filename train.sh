python train_net.py --num-gpus 2\
  --config-file configs/USIS10K/instance-segmentation/dinov2/dinov2_vit_large_bs.yaml \
  --dist-url 'tcp://127.0.0.1:49515' \
  --resume

#####################
#--config-file configs/USIS10K/instance-segmentation/dinov2/dinov2_vit_large_bs.yaml \
#--config-file configs/USIS10K/Class_Agnostic/dinov2/dinov2_vit_large_bs.yaml \
#--config-file configs/UIIS/instance-segmentation/dinov2/dinov2_vit_large_bs.yaml \