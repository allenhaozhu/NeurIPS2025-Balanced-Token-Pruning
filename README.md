# Balanced-Token-Pruning

## Install

### Initialize Enviroment
```bash
conda create -n BTP python=3.10 -y
conda activate BTP
```
### Install LLaVA-v1.5
```bash
git clone https://github.com/haotian-liu/LLaVA.git
cd LLaVA
pip install -e .
pip install transformer==4.40.0
pip install torch==2.3.0
```
### Install Flash-attn
1. download wheel file from https://github.com/Dao-AILab/flash-attention/releases/**flash_attn-2.7.4.post1+cu12torch2.3cxx11abiFALSE-cp310-cp310-linux_x86_64.whl**

2. install the wheel
```bash
 pip install xx.wheel
```

### Install lmms-eval
```bash
cd ..
git clone https://github.com/EvolvingLMMs-Lab/lmms-eval
cd lmms-eval
pip install -e .
pip datasets==3.2.0
```
### Check Enviroment (After the Above Steps)

⚠️ Notice: Make Sure **transformer==4.40.0** 
```bash
pip install transformer==4.40.0
```

⚠️ Notice: Make Sure **torch==2.2.0** or **torch==2.3.0**

```bash
pip install torch==2.3.0
```

⚠️ Notice: Make Sure **numpy==1.26.4** 

```bash
pip install numpy==1.26.4
```

⚠️ Notice: Make Sure **datasets==3.2.0** 
```bash
pip install datasets==3.2.0
```

> ⚠️ **Note:** If the above installation steps report environment dependency issues, you can safely ignore the errors.

## Replace with BTP Implementation

**Replace the original llama implementation with BTP implementation**

**Replace:**

/miniconda3/envs/BTP/lib/python3.10/site-packages/transformers/models/llama/modeling_llama.py

**With Given:**
modeling_llama.py

## Run experiment

```bash
conda activate BTP
```

### MME Result
```bash
CUDA_VISIBLE_DEVICES=0 accelerate launch --num_processes=1 -m lmms_eval --model llava   --model_args pretrained="llava-v1.5-7b"   --tasks mme  --batch_size 1 --log_samples --log_samples_suffix BTP --output_path ./logs/
```

### MMB Result

```bash
CUDA_VISIBLE_DEVICES=0 accelerate launch --num_processes=1 -m lmms_eval --model llava   --model_args pretrained="llava-v1.5-7b"   --tasks mmbench_en  --batch_size 1 --log_samples --log_samples_suffix BTP --output_path ./logs/
```

### POPE Result

```bash
CUDA_VISIBLE_DEVICES=0 accelerate launch --num_processes=1 -m lmms_eval --model llava   --model_args pretrained="llava-v1.5-7b"   --tasks pope  --batch_size 1 --log_samples --log_samples_suffix BTP --output_path ./logs/
```

### GQA Result

```bash
CUDA_VISIBLE_DEVICES=0 accelerate launch --num_processes=1 -m lmms_eval --model llava   --model_args pretrained="llava-v1.5-7b"   --tasks gqa  --batch_size 1 --log_samples --log_samples_suffix BTP --output_path ./logs/
```

### SQA Result

```bash
CUDA_VISIBLE_DEVICES=0 accelerate launch --num_processes=1 -m lmms_eval --model llava   --model_args pretrained="llava-v1.5-7b"   --tasks sqa  --batch_size 1 --log_samples --log_samples_suffix BTP --output_path ./logs/
```

## Results on other LVLMs

Our code will be released after acceptance.