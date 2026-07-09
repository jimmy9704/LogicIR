# LogicIR: Logic Gate Networks for Image Restoration

![logicir_logo](logicir_logo.png)

This repository provides the official implementation of **"LogicIR: Logic Gate Networks for Image Restoration"**.

Our implementation builds upon [difflogic](https://github.com/Felix-Petersen/difflogic). On top of it, we implement the convolutional differentiable logic gate network architecture described in [Convolutional Differentiable Logic Gate Networks](https://arxiv.org/abs/2411.04732).


### 🛠️ Conda Environment Setup

For the tested environment (**PyTorch 2.9.1 + CUDA 12.2**), run:

```shell
conda create -n logicir python=3.10
conda activate logicir
pip install -r requirements.txt
pip install -e . --no-build-isolation
```

This code has also been tested with **PyTorch 1.13.0 + CUDA 11.7**. For this environment, install the CUDA 11.7-specific requirements instead:

```shell
conda create -n logicir python=3.10
conda activate logicir
pip install -r requirements_cu117.txt
pip install -e . --no-build-isolation
```

For additional installation support for `difflogic`, please refer to [INSTALLATION_SUPPORT.md](https://github.com/Felix-Petersen/difflogic/blob/main/INSTALLATION_SUPPORT.md).

### ⚡ Quick Start

You can quickly test LogicIR using the sample images included in this repository.

First, set up the environment and download the model weights from [Google Drive](https://drive.google.com/drive/folders/1gP7vdDuMMQ0SMweg7Ixrm4GedaMSqg8t?usp=sharing). Then place the checkpoints in the expected folder.

Required checkpoint structure:

```text
final_trained_model/
├── logicir_s_denoising_std25.pth
├── logicir_s_r4_denoising_std25.pth
├── logicir_l_denoising_std25.pth
└──logicir_s_r4_denoising_std25_tiny.pth
```

Then run:

```shell
bash scripts/test.sh
```

The restored images, noisy inputs, and ground-truth images will be saved to:

```text
results/
```

For training or full evaluation, please prepare the datasets and checkpoints as described below.

### 📦 Datasets

We use the [BSD dataset](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/) for training. The processed training images can be downloaded from the [DnCNN-PyTorch](https://github.com/SaoYan/DnCNN-PyTorch) repository.

Training images are expected to be placed under:

```text
data/train/
```

Validation or test images are expected to be placed under:

```text
data/
```

<details>
<summary>Expected Set12 data structure</summary>

```text
data/
├── train/
│   ├── test_001.png
│   ├── test_002.png
│   └── ...
└── Set12/
    ├── 01.png
    ├── 02.png
    └── ...
```

</details>


During training, the preprocessing step generates HDF5 files used by the dataloader:

```text
data/train.h5
data/val.h5
```

### ✅ Test

Model weights are available from [Google Drive](https://drive.google.com/drive/folders/1gP7vdDuMMQ0SMweg7Ixrm4GedaMSqg8t?usp=sharing). Please download them and place them in the appropriate folder.

For testing, place the checkpoints as follows:

```text
final_trained_model/
├── logicir_s_denoising_std25.pth
├── logicir_s_r4_denoising_std25.pth
├── logicir_l_denoising_std25.pth
└──logicir_s_r4_denoising_std25_tiny.pth
```

Then run:

```shell
bash scripts/test.sh
```

<details>
<summary>Detailed test commands</summary>

```shell
# LogicIR-S
CUDA_VISIBLE_DEVICES=0 python ./experiments/test.py \
  --noiseL 25 \
  --val_noiseL 25 \
  --batchSize 1 \
  --channels 2048 \
  --test_data Set12 \
  --rotation \
  --test_file final_trained_model/logicir_s_denoising_std25.pth
```

</details>

If you want to evaluate your own checkpoint, replace `--test_file` with the path to your trained `.pth` file.

### 🚀 Train

#### Start Training

To train LogicIR-S for Gaussian denoising with noise level 25, run:

```shell
bash scripts/train.sh
```

Checkpoints and training logs will be saved to:

```text
logs/logicir_s_denoising_std25/
```

<details>
<summary>Detailed training command</summary>

```shell
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
```

</details>

#### Fine-tuning

After the initial training stage, LogicIR is further fine-tuned using the straight-through estimator (STE).

To fine-tune LogicIR-S, run:

```shell
bash scripts/finetune.sh
```

<details>
<summary>Detailed fine-tuning command</summary>

```shell
CUDA_VISIBLE_DEVICES=0 python ./experiments/finetune.py \
  --noiseL 25 \
  --val_noiseL 25 \
  --batchSize 3 \
  --channels 2048 \
  --log_every 100 \
  --lr 1e-3 \
  --implementation 'cuda_ste' \
  --outf logs/finetune_logicir_s_denoising_std25
```

</details>

#### Rotation Training (Optional)

After fine-tuning, you can run the rotation-based training stage to obtain rotation-enhanced models such as the -2RT and -4RT variants.

To run rotation training, use:

```shell
bash scripts/rotation.sh
```

For an RTX 4090, the batch size is set to 1 in the provided script.

<details>
<summary>Detailed rotation training command</summary>

```shell
CUDA_VISIBLE_DEVICES=0 python ./experiments/rotation.py \
  --noiseL 25 \
  --val_noiseL 25 \
  --batchSize 1 \
  --channels 2048 \
  --log_every 50 \
  --lr 1e-3 \
  --outf logs/rotation_logicir_s_denoising_std25_final
```

</details>

### 🔋 FPGA Deployment

We provide a compact `--tiny True` option, referred to as **LogicIR-S-Tiny**, for resource-constrained settings. This lightweight variant replaces the full multi-scale LogicIR-S architecture with a sequential logic-gate network and uses a narrower internal feature width, reducing hardware resource usage while preserving restoration capability.

LogicIR-S-Tiny is particularly suitable for small FPGA devices, where area, memory, and latency are constrained and a favorable performance-efficiency trade-off is important. We thank the [EcoLogic](https://github.com/matheusmaldaner/EcoLogic) project, which provides a useful reference for FPGA implementation.


### 📌 Citation

If this code is useful for your research, please cite our paper.

```bibtex
@inproceedings{lee2026logicir,
  title={LogicIR: Logic Gate Networks for Image Restoration},
  author={Lee, Hongjae and Son, Myungjun and Yu, Jaeseong and Jung, Seung-Won},
  booktitle={European Conference on Computer Vision},
  year={2026}
}
```

### 📬 Contact

Email: [jimmy9704@korea.ac.kr](mailto:jimmy9704@korea.ac.kr)
