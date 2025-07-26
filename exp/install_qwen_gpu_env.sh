#!/bin/bash

# 适配4090 GPU的大模型训练环境一键安装脚本
# 建议在base环境下运行

ENV_NAME=qwen_gpu_env
PYTHON_VERSION=3.11

# 创建conda环境
conda create -n $ENV_NAME python=$PYTHON_VERSION -y

# 激活环境
source $(conda info --base)/etc/profile.d/conda.sh
conda activate $ENV_NAME

# 安装PyTorch+CUDA
conda install pytorch==2.1.0 torchvision==0.16.0 pytorch-cuda=11.8 -c pytorch -c nvidia -y

# 安装数据处理相关包
conda install numpy=1.26.4 pandas=2.1.4 pyarrow=12.0.1 tqdm requests -c conda-forge -y

# 安装大模型生态包
pip install transformers==4.38.2 peft==0.7.1 bitsandbytes==0.43.0 accelerate==0.26.1 datasets==2.13.0 modelscope==1.9.5 evaluate==0.4.1 python-dotenv

# 检查GPU可用性
python -c "import torch; print('torch:', torch.__version__); print('cuda:', torch.cuda.is_available()); print('gpu:', torch.cuda.get_device_name(0))"

echo "\n环境 $ENV_NAME 安装完成！可直接用于Qwen2/LoRA/大模型训练推理。" 