# CrossND - Cross-Domain Name Disambiguation Project

## Overview

CrossND is a cross-domain name disambiguation project based on Large Language Models (LLM), supporting datasets such as KDDCup and WhoisWho.

This project implements the method from the paper **"Cross-Source Reasoning-based Correction for Author Name Disambiguation"**, using LoRA fine-tuning and reasoning-based correction mechanisms for author name disambiguation.

## Project Structure

```
crossnd/
├── config/              # DeepSpeed and training configurations
├── kddcup_data/        # Dataset files
├── train.sh            # Training script
├── inf.sh              # Inference script
├── train.py            # Main training program
├── model.py            # Model definitions
├── trainer.py          # Custom trainer
├── utils.py            # Utility functions
└── inference.py        # Inference utilities
```

## Installation

### 1. Install Dependencies

```bash
cd crossnd
pip install -r requirements.txt
```

Main dependencies:
- PyTorch
- Transformers
- DeepSpeed
- LoRA/PEFT

### 2. Download Data

The KDDCup dataset is available on Hugging Face: [canalpang/kddcup_for_crossnd](https://huggingface.co/datasets/canalpang/kddcup_for_crossnd)

```bash
# Download data using modelscope (optional)
modelscope download --dataset canalpang/crossnd-kddcup --local_dir ./kddcup_data
```

The `kddcup_data/` directory contains all necessary data files for training and evaluation:

- `alldata_nd_thr09_inout_sim.json` - Main training data
- `train_triplets.json`, `test_with_sim.json`, `valid_with_sim.json` - Train/test/validation splits
- `aid_to_pids_in.json`, `aid_to_pids_out.json` - Author-paper mappings
- `paper_dict_mag.json`, `pub_dict.json` - Paper metadata
- Other supporting data files

## Usage

### Training Model

```bash
# Before running, modify the model path in train.sh
# MODEL_PATH="your/Qwen3-8B"

bash train.sh
```

**Training Configuration:**

- **GPU Configuration**: 8 GPUs by default (modify `DEEPSPEED_GPUS` to adjust)
- **LoRA Parameters**: 
  - Rank: 16
  - Alpha: 32
  - Dropout: 0.05
- **Training Parameters**:
  - Epochs: 4
  - Learning Rate: 2e-5
  - Batch Size: 1 per device
  - Gradient Accumulation: 8 steps
- **Loss Type**: PSL v2 (Probabilistic Soft Logic)
- **Output Directory**: `output/psl/`

**Customizing Training Parameters:**

You can modify training parameters by editing variables in `train.sh`:

```bash
# Example: Change number of GPUs
DEEPSPEED_GPUS="localhost:0,1,2,3"

# Example: Change learning rate
LEARNING_RATE=1e-5

# Example: Change output directory
OUTPUT_DIR="output/my_experiment"

# Example: Change model path
MODEL_PATH="Qwen/Qwen3-8B"  # Use Hugging Face model
# or
MODEL_PATH="/path/to/local/Qwen3-8B"  # Use local model
```

### Inference and Evaluation

After training, you can run inference using the inference script:

```bash
# Before running, modify the model path and LoRA checkpoint path in inf.sh
# MODEL_PATH="your/Qwen3-8B"
# LORA_PATH="output/Qwen8B/cls_outer_v4_psl/checkpoint-200"

bash inf.sh
```

**Inference Configuration:**

- **GPU Configuration**: 8 GPUs by default (modify `DEEPSPEED_GPUS` to adjust)
- **Model Path**: Uses `Qwen/Qwen3-8B` by default
- **LoRA Checkpoint**: Loads from `output/Qwen8B/cls_outer_v4_psl/checkpoint-200`
- **Inference Parameters**:
  - Paper Selection Number: 100
  - Number of Turns: 10
  - Batch Size: 1 per device
  - Loss Type: PSL v2
- **Output Directory**: `output/inference/`

**Customizing Inference Parameters:**

You can modify inference parameters by editing variables in `inf.sh`:

```bash
# Example: Change LoRA checkpoint path
LORA_PATH="output/psl/checkpoint-best"

# Example: Change number of GPUs
DEEPSPEED_GPUS="localhost:0,1,2,3"

# Example: Change output directory
OUTPUT_DIR="output/my_inference"

# Example: Change paper selection number
PAPER_SLCT_NUM=50
```

## Key Features

- **LoRA Fine-tuning**: Efficient parameter-efficient fine-tuning method
- **DeepSpeed Integration**: Distributed training with ZeRO optimization
- **Binary Classification Head**: Custom classification layer for disambiguation
- **Outer Product Features**: Enhanced feature representation
- **PSL Loss**: Probabilistic Soft Logic for improved reasoning
- **Multi-turn Reasoning**: Configurable number of reasoning turns

## Notes

- The script uses DeepSpeed ZeRO Stage 1 for memory optimization
- Gradient checkpointing is enabled to reduce memory usage
- Mixed precision training (BF16) is used to accelerate training
- Best model is automatically saved based on AUC_MAP metric
- Make sure to modify `MODEL_PATH` in the scripts to the correct model path before running

## Citation

If you use this code in your research, please cite our paper:

```bibtex
@inproceedings{crossnd2024,
  title={Cross-Source Reasoning-based Correction for Author Name Disambiguation},
}
```

## License

MIT License
