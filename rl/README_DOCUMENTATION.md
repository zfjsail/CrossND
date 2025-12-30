# CrossND LLM Project - Documentation Index

This directory contains comprehensive documentation of the CrossND LLM codebase. Start here to understand the project!

## Documentation Files (Created)

### 1. **QUICK_START.md** (Start here first!)
   - 5-minute project overview
   - Visual explanation of inputs/outputs
   - How multi-turn classification works
   - Running training examples
   - Common issues and solutions
   - **Best for**: Getting started quickly, understanding the big picture

### 2. **CODEBASE_OVERVIEW.md** (Comprehensive reference)
   - Detailed project structure
   - LLM input structure with examples
   - Multi-turn mechanism explained (num_turn)
   - Model architecture details
   - Loss functions documentation
   - Training pipeline flow
   - Evaluation metrics explanation
   - Special tokens reference
   - **Best for**: Deep understanding, implementation details

### 3. **KEY_FILES_REFERENCE.md** (Navigation guide)
   - Core model files location and purpose
   - Data files organization
   - Training scripts location
   - Key classes and methods
   - Parameter reference
   - Data flow diagrams
   - Model architecture flow
   - Debugging tips
   - **Best for**: Finding specific files, understanding code structure

### 4. **README_DOCUMENTATION.md** (This file)
   - Index of all documentation
   - How to use these guides
   - Learning path recommendations

## Reading Recommendations

### I want to... | Read this first
---|---
Understand what this project does in 5 minutes | QUICK_START.md (Section 1-2)
Run training on my own data | QUICK_START.md (Running Training section)
Understand the data format | CODEBASE_OVERVIEW.md (Section 2) + QUICK_START.md (Data Format)
Learn how num_turn works | CODEBASE_OVERVIEW.md (Section 3)
Find a specific file | KEY_FILES_REFERENCE.md (Quick Navigation)
Understand model internals | CODEBASE_OVERVIEW.md (Section 4)
Debug a training issue | QUICK_START.md (Common Issues) + KEY_FILES_REFERENCE.md (Debugging Tips)
Customize the loss function | CODEBASE_OVERVIEW.md (Section 4.3)
Understand evaluation | CODEBASE_OVERVIEW.md (Section 6) + KEY_FILES_REFERENCE.md (Evaluation and Metrics)
Learn model training flow | CODEBASE_OVERVIEW.md (Section 5) + KEY_FILES_REFERENCE.md (Data Flow Diagram)

## Quick Reference

### Core Files Summary
```
model.py (724 lines)
  ├─ Qwen3ForCrossND - Main model class
  ├─ LlamaForCrossND - Alternative model
  └─ 8+ loss functions (ce, ls, kl, ranking, etc.)

utils.py (509 lines)
  ├─ CrossNDDataset - Data loading and multi-turn grouping
  ├─ CrossNDCollator - Batch padding and metadata
  └─ Special tokens definition

trainer.py (1000+ lines)
  ├─ CrossNDTrainer_v2 - Custom trainer
  ├─ Author-level metric calculation
  └─ Evaluation with metadata preservation

train.py (226 lines)
  └─ Main training entry point

scripts/turn/
  ├─ turn1.sh - Single-turn (1 paper per batch)
  ├─ turn10.sh - Multi-turn (10 papers per batch)
  └─ ... (turn2, turn4, turn8, turn16)
```

### Key Concepts
| Concept | Definition | Read More |
|---------|-----------|-----------|
| **num_turn** | Papers grouped and processed together per forward pass | CODEBASE_OVERVIEW.md Section 3 |
| **label_token** | Special token marking classification position | CODEBASE_OVERVIEW.md Section 2.5 |
| **Multi-turn input** | Multiple papers from same author in one prompt | CODEBASE_OVERVIEW.md Section 3.3 |
| **Binary head** | 2-class output [No, Yes] vs full vocabulary | CODEBASE_OVERVIEW.md Section 4.4 |
| **Loss function** | 8+ options for different training objectives | CODEBASE_OVERVIEW.md Section 4.3 |
| **Author-level metrics** | AUC/MAP calculated per author, then averaged | CODEBASE_OVERVIEW.md Section 6.1 |

### Parameters Guide
```
NUM_TURN        = 1, 2, 4, 8, 10, 16      (papers per batch)
LOSS_TYPE       = ce, ls, kl, ce_fl, ranking, psl
USE_OUTER       = true/false               (external sources)
USE_BINARY_HEAD = true/false               (output head type)
LORA_R          = 16, 32                   (LoRA rank)
LABEL_THR       = 0.9                      (soft→hard label threshold)
```

## Project Task

**Scholar-Paper Anomaly Detection**: Classify whether a paper belongs to an author

```
Input: Scholar name + Paper metadata + Background papers
Output: Probability (0-1) that paper belongs to scholar
Goal: Detect mis-assigned papers in author profiles
```

## Data Pipeline

```
JSON Training Data (aid1, aid2, pid, label, soft_label)
  ↓
CrossNDDataset loads & groups by scholar (num_turn)
  ↓
Each sample: scholar context + N papers + label positions
  ↓
Tokenize & pad in CrossNDCollator
  ↓
Model forward pass at label positions
  ↓
Loss computation & gradient updates
  ↓
Evaluation: author-level AUC/MAP metrics
```

## Training Models Available

- **Qwen-3-4B**: Smaller, faster training
- **Qwen-3-8B**: Better quality (in scripts)
- **Llama-7B/13B**: Alternative models

## Loss Functions Available

- **ce**: Cross-entropy (hard labels)
- **ls**: Label smoothing
- **kl**: KL divergence (soft labels)
- **ce_fl**: Cross-entropy + focal
- **ranking**: Margin ranking
- **psl**: Probabilistic soft logic
- Plus 2 combined variants

## Evaluation Metrics

- **AUC**: Area under ROC curve (per author)
- **MAP**: Mean Average Precision (per author)
- **Aggregation**: Macro-average across all authors
- **Filtering**: Skip invalid authors (all pos/neg, >50% pos ratio)

## File Locations

### Code
- Models: `/workspace/pangyunhe/project/crossnd/llm/model.py`
- Data: `/workspace/pangyunhe/project/crossnd/llm/utils.py`
- Training: `/workspace/pangyunhe/project/crossnd/llm/train.py`
- Trainer: `/workspace/pangyunhe/project/crossnd/llm/trainer.py`

### Data
- Training data: `/workspace/pangyunhe/project/crossnd/llm/data/alldata_nd_thr09.json`
- Paper metadata: `/workspace/pangyunhe/project/crossnd/data/.../kddcup/paper_dict_mag.json`
- Author mappings: `/workspace/pangyunhe/project/crossnd/data/.../kddcup/aid_to_pids_*.json`

### Scripts
- Single-turn: `/workspace/pangyunhe/project/crossnd/llm/scripts/turn/turn1.sh`
- Multi-turn: `/workspace/pangyunhe/project/crossnd/llm/scripts/turn/turn10.sh`
- Qwen-8B: `/workspace/pangyunhe/project/crossnd/llm/scripts/qwen8B/cls-hard-outer.sh`

### Outputs
- Checkpoints: `/workspace/pangyunhe/project/crossnd/llm/output/*/checkpoint-X/`
- Results: `/workspace/pangyunhe/project/crossnd/llm/output/*/result.txt`
- W&B logs: `/workspace/pangyunhe/project/crossnd/llm/wandb/`

## Typical Workflows

### Workflow 1: Single-Turn Training (Simplest)
```bash
1. Read QUICK_START.md (5 minutes)
2. bash scripts/turn/turn1.sh
3. Monitor output_dir/result.txt
```

### Workflow 2: Multi-Turn Training (Recommended)
```bash
1. Read QUICK_START.md (5 minutes)
2. bash scripts/turn/turn10.sh
3. Monitor output_dir/result.txt and W&B logs
```

### Workflow 3: Custom Configuration
```bash
1. Read CODEBASE_OVERVIEW.md (15 minutes)
2. Read KEY_FILES_REFERENCE.md (10 minutes)
3. Modify scripts/turn/turn10.sh with custom parameters
4. Run modified script
5. Analyze results in output_dir/result.txt
```

### Workflow 4: Implementation Understanding
```bash
1. Read QUICK_START.md (5 minutes)
2. Read CODEBASE_OVERVIEW.md (30 minutes)
3. Read KEY_FILES_REFERENCE.md (20 minutes)
4. Study model.py, utils.py, trainer.py in code editor
5. Run training with debugging enabled
```

## Learning Path (Recommended)

**For Quick Understanding (15 minutes)**
1. Read QUICK_START.md completely
2. Understand num_turn from CODEBASE_OVERVIEW.md Section 3
3. Ready to run training!

**For Comprehensive Understanding (60 minutes)**
1. Read QUICK_START.md (5 min)
2. Read CODEBASE_OVERVIEW.md Sections 1-6 (30 min)
3. Read KEY_FILES_REFERENCE.md (20 min)
4. Run turn1.sh training (5 min)

**For Implementation Details (2-3 hours)**
1. Complete Comprehensive Understanding path
2. Read model.py and understand loss functions
3. Read utils.py and understand CrossNDDataset
4. Read trainer.py and understand metrics
5. Run debugging code snippets from KEY_FILES_REFERENCE.md

## Quick Answers

**Q: What does num_turn=10 mean?**
A: Process 10 papers from the same scholar simultaneously, sharing context. See CODEBASE_OVERVIEW.md Section 3.

**Q: How is the model input structured?**
A: System prompt + Scholar context (internal/external papers + similarity) + Multiple papers with label positions. See QUICK_START.md "Input Structure".

**Q: What's the label_token?**
A: A special marker showing where to predict Yes/No. See CODEBASE_OVERVIEW.md Section 2.5.

**Q: Which loss function should I use?**
A: Start with 'ce' for hard labels, try 'kl' for soft labels. See QUICK_START.md "Loss Functions".

**Q: How are metrics calculated?**
A: Per-author AUC/MAP, then macro-averaged across authors. See CODEBASE_OVERVIEW.md Section 6.

**Q: How do I modify the code?**
A: Find files in KEY_FILES_REFERENCE.md, understand in CODEBASE_OVERVIEW.md, test with scripts.

## Need Help?

- **Understanding concept**: Check CODEBASE_OVERVIEW.md
- **Finding a file**: Check KEY_FILES_REFERENCE.md
- **Getting started**: Read QUICK_START.md
- **Debugging code**: Check KEY_FILES_REFERENCE.md "Debugging Tips"
- **Running training**: Read QUICK_START.md "Running Training" section

## Documentation Version

Created: 2025-10-29
Project: CrossND LLM Scholar-Paper Anomaly Detection
Codebase location: `/workspace/pangyunhe/project/crossnd/llm/`

---

**Start with QUICK_START.md for a fast introduction, or dive into CODEBASE_OVERVIEW.md for comprehensive details!**

