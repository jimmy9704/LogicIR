# LogicIR-S
CUDA_VISIBLE_DEVICES=0 python ./experiments/test.py \
  --noiseL 25 \
  --val_noiseL 25 \
  --batchSize 1 \
  --channels 2048 \
  --test_data BSD68 \
  --test_file final_trained_model/logicir_s_denoising_std25.pth

# # LogicIR-L
# CUDA_VISIBLE_DEVICES=0 python ./experiments/test.py \
#   --noiseL 25 \
#   --val_noiseL 25 \
#   --batchSize 1 \
#   --channels 4096 \
#   --test_data BSD68 \
#   --test_file final_trained_model/logicir_l_denoising_std25.pth

# # LogicIR-S-4RT
# CUDA_VISIBLE_DEVICES=0 python ./experiments/test.py \
#   --noiseL 25 \
#   --val_noiseL 25 \
#   --batchSize 1 \
#   --channels 2048 \
#   --test_data BSD68 \
#   --rotation \
#   --test_file final_trained_model/logicir_s_r4_denoising_std25.pth

# # LogicIR-S-tiny
# CUDA_VISIBLE_DEVICES=0 python ./experiments/test.py \
#   --noiseL 25 \
#   --val_noiseL 25 \
#   --batchSize 1 \
#   --channels 2048 \
#   --test_data BSD68 \
#   --tiny True \
#   --rotation \
#   --test_file final_trained_model/logicir_s_r4_denoising_std25_tiny.pth
