# CrossND - 交叉域命名消歧项目

## 项目简介

CrossND 是一个基于大语言模型(LLM)的交叉域命名消歧项目，支持 WhoisWho 和 KDDCup 等数据集。

## 安装

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 下载数据
```bash
modelscope login  --token ms-5ef9d8f4-c656-48a4-9af9-b6e660d4ee42
modelscope download --dataset canalpang/crossnd-whoiswho --local_dir ./whoiswho_data
```
## 使用方式

### 训练评估模型

```bash 
bash /workspace/pangyunhe/project/crossnd/crossnd/scripts/whoiswho/psl.sh

bash /workspace/pangyunhe/project/crossnd/crossnd/scripts/whoiswho/psl_lambda0.2_psi1.0.sh

bash /workspace/pangyunhe/project/crossnd/crossnd/scripts/whoiswho/psl_lambda0.5_psi1.0.sh

bash /workspace/pangyunhe/project/crossnd/crossnd/scripts/whoiswho/psl_lambda0.5_psi1.4.sh

bash /workspace/pangyunhe/project/crossnd/crossnd/scripts/whoiswho/psl_lambda0.8_psi1.0.sh
```
