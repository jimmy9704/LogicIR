# LogicIR-S
CUDA_VISIBLE_DEVICES=0 python ./experiments/finetune.py \
  --noiseL 25 \
  --val_noiseL 25 \
  --batchSize 3 \
  --channels 2048 \
  --log_every 100 \
  --lr 1e-3 \
  --implementation 'cuda_ste' \
  --outf logs/finetune_logicir_s_denoising_std25

# # LogicIR-S-tiny
# CUDA_VISIBLE_DEVICES=0 python ./experiments/finetune.py \
#   --noiseL 25 \
#   --val_noiseL 25 \
#   --batchSize 3 \
#   --channels 2048 \
#   --log_every 100 \
#   --lr 1e-3 \
#   --tiny True \
#   --implementation 'cuda_ste' \
#   --outf logs/finetune_logicir_s_denoising_std25_tiny
