# Balanced Token Pruning: Accelerating Vision Language Models Beyond Local Optimization (Neurips 2025)

## 📺 News. 

**[2025.9.25]** 🚀 We release the Qwen-2.5-VL version code!

**[2025.9.18]** 🎉 Our paper has been accepted by Neurips 2025!!!

**[2025.7.25]** 🚀 We release the LLaVA version code!

**[2025.5.28]** 🚀 We release the paper at [ArXiv](https://arxiv.org/abs/2505.22038)!

## 💡 Highlights
- 🔥 **Pruning layer determination**: We determine the pruning layers using a calibration set. No answers are required — we only utilize the image processing procedure.
- 🔥 **Cross-layer pruning analysis**: we analyze how shallow-layer pruning affects deeper layers and propose our balanced token pruning method.
- 🔥 **Support FlashAttention**:
Our codebase support flash_attention_2.


## 👨‍💻 Todo

- [ ] Release the code of llava-next
- [ ] Release all the baseline codes.
- [ ] Further simplify the code to Qwen and series.

## ⚙️ Install

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
download wheel file from https://github.com/Dao-AILab/flash-attention/releases/**flash_attn-2.7.4.post1+cu12torch2.3cxx11abiFALSE-cp310-cp310-linux_x86_64.whl**  
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



## 🚄 Quick Start

### LLaVA Results

**Replace:**

/miniconda3/envs/BTP/lib/python3.10/site-packages/transformers/models/llama/modeling_llama.py

**With Given:**
modeling_llama.py

#### Run experiment

```bash
conda activate BTP
```

#### MME Result
```bash
CUDA_VISIBLE_DEVICES=0 accelerate launch --num_processes=1 -m lmms_eval --model llava   --model_args pretrained="llava-v1.5-7b"   --tasks mme  --batch_size 1 --log_samples --log_samples_suffix BTP --output_path ./logs/
```

#### MMB Result

```bash
CUDA_VISIBLE_DEVICES=0 accelerate launch --num_processes=1 -m lmms_eval --model llava   --model_args pretrained="llava-v1.5-7b"   --tasks mmbench_en  --batch_size 1 --log_samples --log_samples_suffix BTP --output_path ./logs/
```

#### POPE Result

```bash
CUDA_VISIBLE_DEVICES=0 accelerate launch --num_processes=1 -m lmms_eval --model llava   --model_args pretrained="llava-v1.5-7b"   --tasks pope  --batch_size 1 --log_samples --log_samples_suffix BTP --output_path ./logs/
```

#### GQA Result

```bash
CUDA_VISIBLE_DEVICES=0 accelerate launch --num_processes=1 -m lmms_eval --model llava   --model_args pretrained="llava-v1.5-7b"   --tasks gqa  --batch_size 1 --log_samples --log_samples_suffix BTP --output_path ./logs/
```

#### SQA Result

```bash
CUDA_VISIBLE_DEVICES=0 accelerate launch --num_processes=1 -m lmms_eval --model llava   --model_args pretrained="llava-v1.5-7b"   --tasks sqa  --batch_size 1 --log_samples --log_samples_suffix BTP --output_path ./logs/
```




## 🎉 Acknowledgments

- [LLaVA](https://github.com/haotian-liu/LLaVA): the codebase we built upon. Thanks for their brilliant contributions to the community.
- [Open-LLaVA-NeXT](https://github.com/xiaoachen98/Open-LLaVA-NeXT): Thanks for the impressive open-source implementation of LLaVA-NeXT series.
- [Qwen-2.5-VL](https://github.com/QwenLM/Qwen3-VL): Thanks for the impressive open-source implementation of Qwen-2.5-VL series.
- [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval): the amazing open-sourced codebase for evaluating various LVLMs!
- [FastV](https://github.com/pkunlp-icler/FastV): the excellent pruning methods based on attention.
- [PyramidDrop](https://github.com/Cooperx521/PyramidDrop): The excellent pruning method introduces a layer-wise pruning strategy.
- [Divprune](https://github.com/vbdi/divprune): the excellent pruning methods based on diversity.

