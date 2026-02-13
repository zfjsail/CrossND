# CrossND - Cross-Source Reasoning-based Correction for Author Name Disambiguation

This repository contains the implementation of the paper **"Cross-Source Reasoning-based Correction for Author Name Disambiguation"**.

The project implements a cross-source author name disambiguation system using Large Language Models (LLMs) with LoRA fine-tuning and reasoning-based correction mechanisms.

## Branches

- **main**: Contains the KDD Cup dataset implementation
- **whoiswho**: Contains the WhoIsWho dataset implementation (see [whoiswho branch](https://github.com/zfjsail/CrossND/tree/whoiswho))

## Project Structure

```
crossnd/
├── config/              # DeepSpeed and training configurations
├── kddcup_data/        # Dataset files
├── scripts/            # Training scripts
├── train.py            # Main training script
├── model.py            # Model definitions
├── trainer.py          # Custom trainer implementation
├── utils.py            # Utility functions
└── inference.py        # Inference utilities
```

## Dataset

The `kddcup_data/` directory contains all necessary data files for training and evaluation:

- `alldata_nd_thr09_inout_sim.json` - Main training data
- `train_triplets.json`, `test_with_sim.json`, `valid_with_sim.json` - Train/test/validation splits
- `aid_to_pids_in.json`, `aid_to_pids_out.json` - Author-paper mappings
- `paper_dict_mag.json`, `pub_dict.json` - Paper metadata
- Other supporting data files

## Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```

Main dependencies:
- PyTorch
- Transformers
- DeepSpeed
- LoRA/PEFT

## Model Setup

The training script uses the Qwen3-8B model. You have two options:

1. **Use Hugging Face Model Hub** (Recommended for first-time users):
   - The script is configured to use `"Qwen/Qwen3-8B"` by default
   - The model will be automatically downloaded on first run

2. **Use Local Model**:
   - Download the Qwen3-8B model to a local directory
   - Update the `MODEL_PATH` variable in `scripts/train.sh`

## Training

To start training:

```bash
cd crossnd
bash scripts/train.sh
```

### Training Configuration

The training script (`scripts/train.sh`) includes the following key parameters:

- **GPU Configuration**: 8 GPUs by default (modify `DEEPSPEED_GPUS` for different setups)
- **LoRA Parameters**: 
  - Rank: 16
  - Alpha: 32
  - Dropout: 0.05
- **Training Parameters**:
  - Epochs: 10
  - Learning Rate: 2e-5
  - Batch Size: 1 per device
  - Gradient Accumulation: 8 steps
- **Loss Type**: PSL v2 (Probabilistic Soft Logic)
- **Output**: Results saved to `output/Qwen8B/cls_outer_v4_psl/`

### Customization

You can modify training parameters by editing variables in `scripts/train.sh`:

```bash
# Example: Change number of GPUs
DEEPSPEED_GPUS="localhost:0,1,2,3"

# Example: Change learning rate
LEARNING_RATE=1e-5

# Example: Change output directory
OUTPUT_DIR="output/my_experiment"
```

## Inference

After training, use the inference utilities to make predictions:

```python
from inference import load_model, predict

# Load trained model
model = load_model("output/Qwen8B/cls_outer_v4_psl/checkpoint-best")

# Make predictions
results = predict(model, test_data)
```

## Features

- **LoRA Fine-tuning**: Efficient parameter-efficient fine-tuning
- **DeepSpeed Integration**: Distributed training with ZeRO optimization
- **Binary Classification Head**: Custom classification layer for disambiguation
- **Outer Product Features**: Enhanced feature representation
- **PSL Loss**: Probabilistic Soft Logic for improved reasoning
- **Multi-turn Reasoning**: Configurable number of reasoning turns

## Notes

- The script uses DeepSpeed ZeRO Stage 1 for memory optimization
- Gradient checkpointing is enabled to reduce memory usage
- Mixed precision training (BF16) is used for faster training
- Best model is automatically saved based on AUC_MAP metric

## Citation

If you use this code in your research, please cite our paper:

```bibtex
@inproceedings{crossnd2024,
  title={Cross-Source Reasoning-based Correction for Author Name Disambiguation},
  author={Your Name},
  booktitle={Proceedings of KDD Cup},
  year={2024}
}
```

## License

MIT License
