# cd /workspace/pangyunhe/models
# bash download.sh
cd /workspace/pangyunhe/project/crossnd/llm
pip install -r requirements.txt
wandb login 14a5316013f658f8ff2f0771a42ee134919be51b
wandb online
wandb enabled

bash /workspace/pangyunhe/project/crossnd/llm/scripts/qwen8B/cls-hard-outer-hybrid.sh
bash /workspace/pangyunhe/project/crossnd/llm/scripts/qwen8B/cls-softce_v2.sh
bash /workspace/pangyunhe/project/crossnd/llm/scripts/qwen8B/cls-softce-outer-hybrid.sh
bash /workspace/pangyunhe/project/crossnd/llm/scripts/turn/run.sh
