# CrossND LLM - Key Files Reference Guide

## Quick Navigation

### Core Model Files
- **`/workspace/pangyunhe/project/crossnd/llm/model.py`** (724 lines)
  - `Qwen3ForCrossND`: Main model class for Qwen-based classification
  - `LlamaForCrossND`: Alternative model class for Llama models
  - Loss functions: 8+ different loss implementations
  - Binary head mechanism for classification
  - Forward pass with label_token position extraction

- **`/workspace/pangyunhe/project/crossnd/llm/utils.py`** (509 lines)
  - `CrossNDDataset`: Main dataset class with:
    - Multi-turn data grouping (num_turn mechanism)
    - Paper selection and formatting
    - Chat template vs raw text formatting
    - Special token handling
  - `CrossNDCollator`: Batch padding and metadata preservation
  - Special token definitions and constants

- **`/workspace/pangyunhe/project/crossnd/llm/trainer.py`** (1000+ lines)
  - `CrossNDTrainer_v2`: Custom trainer extending HuggingFace Trainer
  - `calculate_author_metrics()`: Author-level metric aggregation
  - `cal_auc_map()`: AUC and MAP computation
  - `DataArguments`, `ModelArguments`: Configuration classes
  - Custom evaluation with metadata preservation

### Training & Processing
- **`/workspace/pangyunhe/project/crossnd/llm/train.py`** (226 lines)
  - Main training entry point
  - Model initialization (Qwen3 vs Llama detection)
  - Special token registration
  - Dataset loading and trainer setup
  - LoRA configuration

- **`/workspace/pangyunhe/project/crossnd/llm/process_llm.py`** (108 lines)
  - Data preprocessing for training
  - Build triplets from paper-author pairs
  - Maps between internal and external author datasets

### Configuration
- **`/workspace/pangyunhe/project/crossnd/llm/config/ds_zero_1.json`**
  - DeepSpeed Stage 1 configuration
  - Gradient accumulation, optimization settings
  - Used for distributed training with 4-8 GPUs

## Data Files Location

### Training Data
- `/workspace/pangyunhe/project/crossnd/llm/data/alldata_nd_thr09.json` - Primary training data
- `/workspace/pangyunhe/project/crossnd/llm/data/alldata_nd_thr092.json` - Alternative threshold
- `/workspace/pangyunhe/project/crossnd/llm/data/alldata_crossnd.json` - Cross-domain variant
- `/workspace/pangyunhe/project/crossnd/llm/data/fuzzyneg.json` - Fuzzy negative samples

### Dataset Sources
- `/workspace/pangyunhe/project/crossnd/data/datasets--canalpang--crossnd/snapshots/.../kddcup/`
  - `aid_to_pids_in.json` - Internal author-paper mappings
  - `aid_to_pids_out.json` - External author-paper mappings
  - `paper_dict_mag.json` - Paper metadata

## Training Scripts Location

### Multi-turn Training
- `/workspace/pangyunhe/project/crossnd/llm/scripts/turn/turn1.sh` - Single paper
- `/workspace/pangyunhe/project/crossnd/llm/scripts/turn/turn10.sh` - 10 papers per batch
- `/workspace/pangyunhe/project/crossnd/llm/scripts/turn/turn16.sh` - 16 papers per batch

### Model-Specific Scripts
- `/workspace/pangyunhe/project/crossnd/llm/scripts/qwen8B/cls-hard-outer.sh`
- `/workspace/pangyunhe/project/crossnd/llm/scripts/qwen8B/cls-softce.sh`
- `/workspace/pangyunhe/project/crossnd/llm/scripts/qwen8B/cls-hard-outer-ranking.sh`

### Dataset-Specific Scripts
- `/workspace/pangyunhe/project/crossnd/llm/scripts/whoiswho/` - WhoisWho dataset configs

## Output Locations

### Model Checkpoints
- `/workspace/pangyunhe/project/crossnd/llm/output/cls-qwen8b-alldatand09_inner/` - Binary head variant
- `/workspace/pangyunhe/project/crossnd/llm/output/cls-multiturn-hybriddata/` - Multi-turn training
- `/workspace/pangyunhe/project/crossnd/llm/output/Qwen8B/` - Qwen 8B experiments

### Training Logs
- W&B runs in: `/workspace/pangyunhe/project/crossnd/llm/wandb/`
- Latest run: `/workspace/pangyunhe/project/crossnd/llm/wandb/latest-run`

## Key Classes and Methods

### CrossNDDataset
```python
Location: utils.py, lines 68-407
Key methods:
  __init__()        - Load data, apply num_turn grouping
  __getitem__(idx)  - Return tokenized sample with metadata
  __len__()         - Return dataset size
  _fetch_single_paper_input() - Format paper as text
  _select_related_papers()    - Sample papers from sources
```

### Qwen3ForCrossND
```python
Location: model.py, lines 78-490
Key methods:
  forward()              - Main forward pass
  compute_loss()         - Route to appropriate loss function
  add_special_tokens()   - Register special tokens
  monkey_patch_cls_head() - Replace vocab output with binary head
  soft_ce_loss()        - Soft label cross-entropy
  kl_divergence_loss()  - KL divergence for soft labels
  margin_ranking_loss() - Ranking loss
```

### CrossNDTrainer_v2
```python
Location: trainer.py, lines 599+
Key methods:
  evaluate()                 - Custom evaluation with author-level metrics
  get_logits()              - Extract logits at label positions
  _remove_unused_columns()  - Preserve metadata in pipeline
```

## Important Parameters

### In training scripts:
```bash
NUM_TURN=10              # Papers per batch (1, 2, 4, 8, 10, 16)
LOSS_TYPE="ce"           # ce, ls, kl, ce_fl, ranking, psl, etc.
USE_OUTER="true"         # Include external source papers
USE_BINARY_HEAD="true"   # Binary classification head
PAPER_SLCT_NUM=100       # Max papers from sources
LABEL_THR=0.9            # Soft label threshold
LORA_R=16                # LoRA rank
GRADIENT_ACCUMULATION=8  # Steps before parameter update
```

### In code (model.py):
```python
self.loss_type              # Selected loss function
self.is_binary_head         # Whether using 2-class output
self.LABEL_TOKEN_IDS        # Position marker token ID
self.YES_TOKEN_IDS          # "Yes" token ID
self.NO_TOKEN_IDS           # "No" token ID
```

### In code (utils.py):
```python
LABEL_TOKEN = '<label_token>'   # Position for classification
YES_TOKEN = 'Yes'               # Positive class token
NO_TOKEN = 'No'                 # Negative class token
paper_slct_num = 100            # Default papers to sample
```

## Data Flow Diagram

```
Raw Data (JSON)
    ↓
CrossNDDataset.__init__()
    ├─ Load JSON file
    ├─ Group by author (if num_turn > 1)
    ├─ Shuffle samples
    └─ Create (num_turn) sized groups
    ↓
CrossNDDataset.__getitem__()
    ├─ Get papers list from group
    ├─ Fetch paper details from paper_dict
    ├─ Select inner papers (random sampling)
    ├─ Select outer papers (random sampling)
    ├─ Format into prompt:
    │  ├─ System prompt
    │  ├─ Global context (name, inner, outer, sim)
    │  └─ Per-paper prompts (with label_token)
    ├─ Tokenize
    └─ Return {input_ids, attention_mask, labels, metadata}
    ↓
CrossNDCollator()
    ├─ Pad variable-length sequences
    ├─ Preserve metadata
    └─ Return batch tensors
    ↓
Model.forward()
    ├─ Pass through transformer
    ├─ Get lm_logits [batch, seq_len, vocab]
    ├─ Extract logits at label_token positions
    ├─ Compute loss or return scores
    └─ Update gradients
```

## Model Architecture Flow

```
Input: text with <label_token> markers
    ↓
Tokenizer.apply_chat_template() or encode_plus()
    ├─ Converts text → token IDs
    ├─ Marks <label_token> positions
    └─ Returns input_ids, attention_mask
    ↓
Model.forward(input_ids, attention_mask, labels, metadata)
    ├─ Qwen3Model / LlamaModel (backbone)
    │  └─ Returns hidden_states [batch, seq_len, hidden_size]
    ├─ lm_head (nn.Linear)
    │  └─ Projects to vocab: [batch, seq_len, vocab_size]
    ├─ Extract label positions
    │  ├─ Find indices where token == LABEL_TOKEN_ID
    │  └─ Get logits[:, indices-1, :]  (shift by -1)
    ├─ Compute loss (if labels provided)
    │  └─ Routes to loss_type specific function
    ├─ Extract score
    │  └─ softmax([YES_logit, NO_logit])[1]
    └─ Return INDModelWithPast(loss, logits, hidden_states)
```

## Typical Workflow

### 1. Single-Turn Training
```bash
bash /workspace/pangyunhe/project/crossnd/llm/scripts/turn/turn1.sh
# Processes 1 paper per forward pass
# Lower memory, simpler execution
```

### 2. Multi-Turn Training
```bash
bash /workspace/pangyunhe/project/crossnd/llm/scripts/turn/turn10.sh
# Processes 10 papers per forward pass
# Better context, higher throughput
```

### 3. With External Data
```bash
# In script: --use_outer true
# Loads both internal and external source papers
# Useful for cross-domain scenarios
```

### 4. Different Loss Functions
```bash
# In script: --loss_type {ce|ls|kl|ce_fl|ranking|psl}
# ce: standard cross-entropy
# ls: label smoothing
# kl: KL divergence (soft labels)
# ce_fl: cross-entropy + focal loss
# ranking: margin ranking
# psl: probabilistic soft logic
```

## Evaluation and Metrics

### Metrics Computed
- **AUC**: Area under ROC curve (per author)
- **MAP**: Mean Average Precision (per author)
- **Final Metric**: Macro-averaged AUC and MAP across all authors

### Metric Calculation (trainer.py, lines 123-212)
1. Group predictions by author ID
2. For each author:
   - Skip if all positive/negative or >50% positive ratio
   - Calculate ROC-AUC and Average Precision
3. Macro-average across valid authors
4. Save to `output_dir/result.txt`

## Debugging Tips

### Check Multi-Turn Grouping
```python
# In utils.py, after data grouping:
print(f"Number of samples: {len(self.data)}")
print(f"Sample 0 papers: {len(self.data[0])}")
```

### Monitor Label Token Positions
```python
# In model.py forward():
indices = (input_ids.squeeze(0) == self.LABEL_TOKEN_IDS).nonzero()
print(f"Label positions: {indices.squeeze(-1)}")
```

### Check Metadata Flow
```python
# In trainer.py evaluate():
print(f"Metadata[0]: {inputs['metadata'][0][0]}")
```

### Verify Loss Computation
```python
# In model.py compute_loss():
print(f"Loss type: {self.loss_type}")
print(f"Logits shape: {logits.shape}")
print(f"Labels shape: {labels.shape}")
```

