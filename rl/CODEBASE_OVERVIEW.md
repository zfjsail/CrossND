# CrossND LLM Project - Comprehensive Codebase Overview

## 1. Project Directory Structure

```
/workspace/pangyunhe/project/crossnd/llm/
├── model.py                 # Core model classes (Qwen3ForCrossND, LlamaForCrossND)
├── train.py                 # Main training script with argument parsing
├── trainer.py               # Custom trainer classes and evaluation logic
├── utils.py                 # Dataset, collator, and utility functions
├── process_llm.py           # Data preprocessing pipeline
├── process.py               # Legacy data processing
├── config/                  # DeepSpeed configuration files
│   ├── ds_zero_1.json
│   ├── ds_zero_2.json
│   └── ds_zero_3.json
├── scripts/                 # Training scripts for various configurations
│   ├── qwen8B/             # Qwen 8B model training scripts
│   ├── turn/               # Multi-turn training scripts (turn1-16)
│   ├── whoiswho/           # WhoisWho dataset specific scripts
│   ├── ablation/           # Ablation study scripts
│   └── old/                # Legacy scripts
├── data/                    # Training data files
│   ├── kddcup/             # KDDCup dataset files
│   ├── whoiswho/           # WhoisWho dataset files
│   ├── alldata_nd.json
│   ├── alldata_nd_thr09.json
│   ├── alldata_nd_thr092.json
│   ├── alldata_crossnd.json
│   ├── fuzzyneg.json
│   └── (other processed data files)
├── output/                  # Training outputs and checkpoints
│   ├── Qwen8B/
│   ├── cls-qwen8b-alldatand09_inner/
│   ├── cls-multiturn-hybriddata/
│   └── (many other model outputs)
├── bin/                     # Evaluation and utility scripts
├── mind/                    # Guard/monitoring modules
└── readme.md                # Quick start guide
```

## 2. LLM Input Structure

### 2.1 Data Format - Training Sample

Each training sample has the following structure:
```json
{
  "aid1": "826",           # Author ID (internal dataset)
  "aid2": "548f27a3dabfaef989f09800",  # Author ID (external dataset, for cross-domain)
  "pid": "2100381566-1",   # Paper ID
  "name": "emanuele_buratti",  # Author name
  "author_sim": 0.839,     # Similarity between internal and external author papers
  "label": 1,              # Hard label (0 or 1)
  "soft_label": 0.97,      # Soft label from external models (0-1)
  "ori_label": -1          # Original label before threshold filtering
}
```

### 2.2 Global Information (System Prompt)

The system prompt defines the task:
```
When using outer data (use_outer=True):
"你要进行一个学者论文检测的任务, 每个学者有两个源可以使用, 一个是内部源, 
另一个是外部源, 使用外部源来支持内部源论文的错误分配检测..."

When NOT using outer data (use_outer=False):
"你要进行一个学者论文检测的任务,每篇论文包题目,学者,机构信息,现在需要
进行异常分配的论文检测..."
```

### 2.3 Global Context Format

```
目标学者名字: {name}
内部源: {selected_inner_papers}  (up to N papers formatted as title+authors+affiliation)
外部源: {selected_outer_papers}  (up to 100 papers)
内部源和外部源的相似度是: {author_sim}
```

### 2.4 Target Paper (Per Label Token)

```
{paper} 是该学者的论文吗?  (when using chat_template)
OR
论文 {paper} 是该学者的论文。 <label_token>  (multi-turn format)
```

Each paper contains:
```
标题: {title}
学者: {author1}, {author2}, ...
机构: {affiliation1}, {affiliation2}, ...
```

### 2.5 Label Token Placement

The special token `<label_token>` marks where the model should predict Yes/No:
- Position is recorded during tokenization
- Labels are shifted by -1 (autoregressive LM prediction)
- At inference time, logits at this position are extracted for classification

## 3. Multi-Round Classification (num_turn Mechanism)

### 3.1 What is num_turn?

`num_turn` is a parameter that controls **multi-paper batch processing**:
- **num_turn=1**: Single paper per sample (batch of 1 paper)
- **num_turn=10**: Group 10 papers (from same author) into one input sequence

### 3.2 How Multi-Turn Works

**Data Grouping in Dataset __init__:**
```python
# Lines 197-214 in utils.py
if model_args.num_turn > 1:
    data_dd = defaultdict(list)
    for item in all_data:
        aid1 = item['aid1']
        data_dd[f'{aid1}'].append(item)  # Group by author
    
    for k, v in data_dd.items():
        for i in range(0, len(v), self.num_turn):
            data.append(v[i:i+self.num_turn])  # Chunk into num_turn sized groups
else:
    self.data = [[i] for i in all_data]  # Single item per group
```

**Input Construction in __getitem__:**
```python
# Lines 220-328 in utils.py
data = self.data[idx]  # This is a list of num_turn items
labels = [i['label'] for i in data]  # Extract labels from all items

for paper in papers:  # All papers in this group
    input += self.multi_turn_prompt.format(
        paper=self._fetch_single_paper_input(paper),
        label_token=LABEL_TOKEN
    )
```

### 3.3 Multi-Turn Input Format

For num_turn=3, the input looks like:
```
<system_prompt>

目标学者名字: {name}
内部源: {papers...}
外部源: {papers...}

论文1 {title+authors+affiliation} 是该学者的论文。<label_token>
论文2 {title+authors+affiliation} 是该学者的论文。<label_token>
论文3 {title+authors+affiliation} 是该学者的论文。<label_token>
```

### 3.4 Multi-Turn Label Processing

Labels are created as a 1D tensor:
```python
labels = torch.tensor([label1, label2, label3], dtype=torch.long)
# Shape: (num_turn,)
# In collator, reshaped to: (batch_size, num_turn) for batching
```

During forward pass:
- Model produces logits for each label_token position
- Loss is computed for all label positions simultaneously
- Evaluation averages metrics across the num_turn samples

### 3.5 Training Dynamics with num_turn

**Effective Batch Size:**
- Physical batch_size=1, num_turn=10 → Process 10 papers per forward pass
- Gradient accumulation multiplies this further
- Example: batch_size=1, num_turn=10, grad_accum=8 → 80 papers before update

**Benefits:**
1. Context from multiple papers of same author in one forward pass
2. More efficient GPU utilization
3. Author-consistent grouping improves coherence

## 4. Model Architecture

### 4.1 Qwen3ForCrossND Class (model.py lines 78-490)

**Key Components:**
```python
class Qwen3ForCrossND(Qwen3PreTrainedModel, GenerationMixin):
    - self.model: Qwen3Model (base transformer)
    - self.lm_head: nn.Linear(hidden_size → vocab_size)
    - self.is_binary_head: Boolean for classification head style
    - self.loss_type: Loss function type selector
```

**Special Tokens:**
```python
self.LABEL_TOKEN_IDS    # Position marker for labels
self.YES_TOKEN_IDS      # Token ID for "Yes"
self.NO_TOKEN_IDS       # Token ID for "No"
```

### 4.2 Forward Pass Logic (lines 400-477)

```
Input: input_ids, attention_mask, labels, metadata
  ↓
Transformer backbone: hidden_states = self.model(...)
  ↓
LM Head: lm_logits = self.lm_head(hidden_states)
  Shape: [batch, seq_len, vocab_size]
  ↓
Extract label positions:
  indices = (input_ids == LABEL_TOKEN_IDS).nonzero()
  logits = lm_logits[:, indices-1, :]  # Shift by -1 (autoregressive)
  ↓
Compute loss (if labels provided):
  loss = self.compute_loss(logits, labels, ...)
  ↓
Extract scores for inference:
  score = softmax([yes_logit, no_logit])[yes_probability]
  ↓
Return: INDModelWithPast(loss, logits=score, ...)
```

### 4.3 Loss Functions (model.py lines 112-396)

**Available Loss Types:**
- `'ce'`: Cross-entropy loss (default)
- `'ls'`: Soft label / label smoothing loss
- `'kl'`: KL divergence loss for soft labels
- `'ce_temperature'`: Temperature-scaled cross-entropy
- `'ce_fl'`: Cross-entropy + focal loss
- `'ranking'`: Margin ranking loss
- `'psl'`: Probabilistic soft logic loss
- Combined: `'ls_ranking'`, `'kl_ranking'`, `'ce_ranking'`

**Custom Loss: soft_ce_loss (for soft labels)**
```python
# Lines 154-182
Input: logits [batch_size, 2], soft_labels [batch_size] ∈ [0,1]
Process:
  - Create soft target distribution: [1-label, label]
  - Apply log_softmax to logits
  - Compute cross-entropy: -sum(soft_target * log_probs)
```

### 4.4 Binary Head Mode

When `use_binary_head=True`:
```python
# Lines 501-511
Instead of vocab_size output:
  binary_head = nn.Linear(hidden_size, 2)
  binary_head.weight[0] = lm_head.weight[NO_TOKEN_ID]
  binary_head.weight[1] = lm_head.weight[YES_TOKEN_ID]

At inference:
  score = softmax(logits)[1]  # Direct probability of "Yes"
```

## 5. Training Pipeline

### 5.1 Training Script Flow (train.py)

```
main()
  ├─ Parse arguments (DataArguments, ModelArguments, TrainingArguments)
  ├─ Load model and tokenizer
  ├─ Add special tokens: <label_token>, <emb_token>, <graph_token>, <eot>, <eog>, <eoe>
  ├─ Resize embeddings to accommodate new tokens
  ├─ Load datasets:
  │  ├─ train_dataset = CrossNDDataset(mode='train')
  │  ├─ eval_dataset = CrossNDDataset(mode='eval')
  │  └─ test_dataset = CrossNDDataset(mode='test')
  ├─ Configure LoRA
  ├─ Create trainer with custom CrossNDTrainer_v2
  └─ trainer.train() → trainer.predict(test_dataset)
```

### 5.2 Dataset Loading (utils.py lines 68-215)

**Mode='train':**
- Load from `model_args.src` (JSON file path)
- Optional: filter by soft_label threshold
- Shuffle and group by author
- Apply num_turn grouping

**Mode='eval' or 'test':**
- Load from predefined validation/test JSON files
- Files vary by dataset (kddcup vs whoiswho)
- Include similarity scores

### 5.3 Data Collator (utils.py lines 410-466)

```python
Processes variable-length sequences:
  - Pad input_ids to max_length with pad_token_id
  - Pad attention_mask with zeros
  - Pad labels to max_length with -100 (ignore_index)
  - Preserve metadata for evaluation
  
Returns:
  {
    'input_ids': [batch_size, max_length],
    'attention_mask': [batch_size, max_length],
    'labels': [batch_size, max_length],
    'metadata': list of metadata dicts
  }
```

### 5.4 Training Arguments Example (scripts/turn/turn10.sh)

```bash
# Model and data
MODEL_PATH="/workspace/pangyunhe/models/Qwen/Qwen3-8B"
DATA_SRC="/workspace/pangyunhe/project/crossnd/llm/data/alldata_nd_thr09.json"
DATA_DIR="/workspace/pangyunhe/project/crossnd/data/.../kddcup"

# Hyperparameters
NUM_TURN=10
LOSS_TYPE="ce"
NUM_EPOCHS=10
LEARNING_RATE=2e-5
TRAIN_BATCH_SIZE=1
GRADIENT_ACCUMULATION=8
EVAL_STEPS=0.1
SAVE_STEPS=0.1

# LoRA
LORA_R=16
LORA_ALPHA=32
LORA_DROPOUT=0.05

# Distributed training
deepspeed --master_port 29500 --include localhost:0,1,2,3,4,5,6,7 train.py \
  --use_outer true \
  --use_binary_head true \
  --num_turn $NUM_TURN \
  --loss_type $LOSS_TYPE \
  --deepspeed "config/ds_zero_1.json" \
  --metric_for_best_model AUC_MAP \
  --bf16
```

## 6. Evaluation Metrics

### 6.1 Metric Computation (trainer.py lines 123-212)

**Author-Level Aggregation:**
```python
for author_id in authors:
    author_probs = get_predictions_for_author(author_id)
    author_labels = get_labels_for_author(author_id)
    
    # Calculate per-author metrics
    author_ap = average_precision_score(author_labels, author_probs)
    author_auc = roc_auc_score(author_labels, author_probs)
    
    # Macro averaging across authors
    final_auc += author_auc / num_authors
    final_map += author_ap / num_authors
```

**Filtering Criteria:**
- Skip authors with all positive or all negative samples
- Skip authors where positive ratio ≥ 50%
- Require at least 2 unique labels per author

### 6.2 Custom Trainer (trainer.py lines 599+)

**CrossNDTrainer_v2:**
- Extends HuggingFace Trainer
- Custom `evaluate()` method with author-level aggregation
- Preserves metadata through evaluation pipeline
- Outputs AUC and MAP metrics
- Saves results to `result.txt`

## 7. Key Configuration Parameters

### DataArguments (trainer.py lines 934-950)
- `data_dir`: Path to dataset directory
- `apply_chat_template`: Use chat format vs raw text
- `dataset`: "kddcup" or "whoiswho"

### ModelArguments (trainer.py lines 953-1020)
- `src`: Training data JSON path
- `model_path`: Pretrained model path
- `lora_r`, `lora_alpha`, `lora_dropout`: LoRA config
- `target_modules`: ["q_proj", "k_proj", "v_proj", "o_proj"]
- `loss_type`: Loss function choice
- `use_binary_head`: Binary classification head
- `use_outer`: Include external source data
- `num_turn`: Number of papers per batch
- `paper_slct_num`: Number of papers to select from sources (default 100)
- `label_thr`: Soft label threshold for hard label conversion
- `author_sim`: Similarity threshold for hybrid training

### TrainingArguments (HuggingFace standard)
- `output_dir`: Checkpoint save path
- `num_train_epochs`: 10 (typical)
- `per_device_train_batch_size`: 1
- `per_device_eval_batch_size`: 1
- `learning_rate`: 2e-5
- `warmup_ratio`: 0.1
- `metric_for_best_model`: "AUC_MAP"
- `save_strategy`, `eval_strategy`: "steps"
- `deepspeed`: DeepSpeed config file

## 8. Special Tokens

**Defined in utils.py (lines 13-27):**
```python
LABEL_TOKEN = '<label_token>'      # Marks classification position
EMBED_TOKEN = '<emb_token>'        # Placeholder (future use)
GRAPH_TOKEN = '<graph_token>'      # Placeholder (future use)
END_OF_TEXT = '<eot>'              # Text end marker
END_OF_GRAPH = '<eog>'             # Graph end marker
END_OF_EMB = '<eoe>'               # Embedding end marker

TRAINABLE_SPECIAL_TOKENS = [END_OF_TEXT, END_OF_GRAPH, END_OF_EMB, LABEL_TOKEN]
additional_special_tokens = [...] + [EMBED_TOKEN, GRAPH_TOKEN]
```

These are added to tokenizer via:
```python
tokenizer.add_special_tokens(special_token_dict)
model.resize_token_embeddings(len(tokenizer))
```

## 9. Data Pipeline Example

### Training Data Preparation
1. Raw data: `alldata_nd_thr09.json` - 4+ fields (aid1, aid2, pid, name, label, soft_label)
2. Dataset loads and groups by author if num_turn > 1
3. For each sample, retrieves:
   - Target papers from `paper_dict_mag.json`
   - Inner source papers from `aid_to_pids_in.json`
   - Outer source papers from `aid_to_pids_out.json`
4. Selects up to 100 papers from each source
5. Formats into chat or multi-turn prompt
6. Tokenizes and pads in collator
7. Batches with label metadata

## 10. File Organization Summary

| Component | Files | Purpose |
|-----------|-------|---------|
| Core Models | model.py | Qwen3/Llama model wrappers with loss functions |
| Training | train.py | Main training entry point |
| Trainer | trainer.py | Custom trainer, evaluation, metrics |
| Data | utils.py | Dataset, collator, special tokens |
| Config | config/ds_zero_*.json | DeepSpeed distributed training configs |
| Scripts | scripts/ | Shell scripts for various training scenarios |
| Data | data/ | JSON training/eval files |
| Output | output/ | Model checkpoints and training outputs |

## 11. Common Training Variants

**Single-turn (num_turn=1):**
- One paper per batch
- Simpler, lower GPU memory

**Multi-turn (num_turn=10):**
- 10 papers grouped from same author
- Better context, higher throughput
- More GPU memory usage

**With/Without Outer:**
- `use_outer=True`: Uses both internal and external source papers
- `use_outer=False`: Uses only internal source papers

**Loss Functions:**
- `ce` (cross-entropy): Standard hard label loss
- `ls` (label smoothing): Smooth distributions
- `kl` (KL divergence): Soft label matching
- `ce_fl` (focal loss): Handle class imbalance

