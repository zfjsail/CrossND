# CrossND Pipeline

This folder contains the three-step pipeline for CrossND (Cross-source Name Disambiguation) on the WhoIsWho dataset.

## Pipeline Overview

```
Step 1: self_clean.py      — Self-cleaning of name_aid_to_pids_in / out
Step 2: refine_crossnd.py  — LLM-based cross-source anomaly detection
Step 3: batch_refine_v2.py — Batch re-scoring with full context
```

## Setup

### 1. Configure LLM API

Edit `LLM.py` and fill in your API credentials in `_call_openai_compatible`:

```python
  API_KEY="YOUR_API_KEY",
  BASE_URL="YOUR_BASE_URL",
```

The code uses an OpenAI-compatible interface. You can point it to any provider (OpenAI, DeepSeek, Qwen, etc.) that supports this protocol.

### 2. Prepare Data

Download the WhoIsWho dataset and place the following files in your data directory:

| File | Description |
|------|-------------|
| `pub_dict.json` | Paper metadata dictionary (pid → paper info) |
| `name_aid_to_pids_in.json` | Internal source: author id → paper ids |
| `name_aid_to_pids_out.json` | External source: author id → paper ids |
| `triplets_train_author_subset.json` | Training triplets |
| `eval_na_checking_triplets_test.json` | Test triplets |
| `eval_na_checking_triplets_valid.json` | Validation triplets |

## Usage

### Step 1 — Self Clean

Clean the author-paper assignment noise from both internal and external sources.

```bash
python self_clean.py \
  --model your-model-name \
  --target /path/to/name_aid_to_pids_in.json \
  --paper_dict /path/to/pub_dict.json \
  --triplets_train /path/to/triplets_train_author_subset.json \
  --triplets_test /path/to/eval_na_checking_triplets_test.json \
  --triplets_valid /path/to/eval_na_checking_triplets_valid.json
```

Run again with `--target /path/to/name_aid_to_pids_out.json` for the external source.

Output is saved to `self_clean/` directory.

### Step 2 — Refine CrossND

LLM-based cross-source anomaly detection on triplet data.

```bash
python refine_crossnd.py \
  --src /path/to/triplets.json \
  --model your-model-name \
  --save_name run_v1 \
  --paper_dict /path/to/pub_dict.json \
  --in_name2pid /path/to/name_aid_to_pids_in.json \
  --out_name2pid /path/to/name_aid_to_pids_out.json \
  --clean_in_name2pid /path/to/self_clean/cleaned_name_aid_to_pids_in.json \
  --clean_out_name2pid /path/to/self_clean/cleaned_name_aid_to_pids_out.json
```

Output is saved to `refine_output/` directory.

### Step 3 — Batch Refine V2

Re-score all papers with full cross-batch context using Step 2 output.

```bash
python batch_refine_v2.py \
  --src /path/to/refine_output/crossnd_xxx.json \
  --model your-model-name \
  --save_name run_v2 \
  --paper_dict /path/to/pub_dict.json \
  --in_name2pid /path/to/name_aid_to_pids_in.json \
  --out_name2pid /path/to/name_aid_to_pids_out.json \
  --clean_in_name2pid /path/to/self_clean/cleaned_name_aid_to_pids_in.json \
  --clean_out_name2pid /path/to/self_clean/cleaned_name_aid_to_pids_out.json
```

Output includes both a full JSON and a flattened per-paper result file in `refine_output/`.

## Dependencies

```bash
pip install openai tqdm
```
