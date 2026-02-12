# CrossND - 交叉域命名消歧项目

## 项目简介

CrossND 是一个基于大语言模型(LLM)的交叉域命名消歧项目，支持 WhoisWho 和 KDDCup 等数据集。

## 安装

### 1. 安装依赖

```bash

cd ./crossnd
pip install -r requirements.txt
```

### 2. 下载数据
```bash
modelscope download --dataset canalpang/crossnd-whoiswho --local_dir ./whoiswho_data
```
## 使用方式

### 训练评估模型

```bash 

# 运行前改动每个sh文件下的 MODEL_PATH="/workspace/pangyunhe/models/Qwen/your_model_path"

bash /workspace/pangyunhe/project/crossnd/crossnd/scripts/whoiswho/psl.sh

bash /workspace/pangyunhe/project/crossnd/crossnd/scripts/whoiswho/psl_lambda0.2_psi1.0.sh

bash /workspace/pangyunhe/project/crossnd/crossnd/scripts/whoiswho/psl_lambda0.5_psi1.0.sh

bash /workspace/pangyunhe/project/crossnd/crossnd/scripts/whoiswho/psl_lambda0.5_psi1.4.sh

bash /workspace/pangyunhe/project/crossnd/crossnd/scripts/whoiswho/psl_lambda0.8_psi1.0.sh
```
