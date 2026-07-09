# LogicIR-S-4RT # batchsize 1 for 4090
CUDA_VISIBLE_DEVICES=0 python ./experiments/rotation.py \
  --noiseL 25 \
  --val_noiseL 25 \
  --batchSize 1 \
  --channels 2048 \
  --log_every 50 \
  --lr 1e-3 \
  --outf logs/rotation_logicir_s_denoising_std25

# # LogicIR-S-tiny
# CUDA_VISIBLE_DEVICES=0 python ./experiments/rotation.py \
#   --noiseL 25 \
#   --val_noiseL 25 \
#   --batchSize 3 \
#   --channels 2048 \
#   --log_every 50 \
#   --lr 1e-3 \
#   --tiny True \
#   --outf logs/rotation_logicir_s_denoising_std25_tiny