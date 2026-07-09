# LogicIR-S
CUDA_VISIBLE_DEVICES=0 python ./experiments/train.py \
  --noiseL 25 \
  --val_noiseL 25 \
  --batchSize 4 \
  --channels 2048 \
  --milestone 35000 \
  --log_every 200 \
  --lr 1e-2 \
  --preprocess True \
  --outf logs/logicir_s_denoising_std25

# # LogicIR-S-tiny
# CUDA_VISIBLE_DEVICES=0 python ./experiments/train.py \
#   --noiseL 25 \
#   --val_noiseL 25 \
#   --batchSize 8 \
#   --channels 2048 \
#   --milestone 35000 \
#   --log_every 200 \
#   --lr 1e-2 \
#   --preprocess True \
#   --tiny True \
#   --outf logs/logicir_s_denoising_std25_tiny
