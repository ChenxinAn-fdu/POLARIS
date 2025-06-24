<div align="center">

#  POLARIS

<div>
   🌠 A <strong>PO</strong>st-training recipe for scaling R<strong>L</strong> on <strong>A</strong>dvanced <strong>R</strong>eason<strong>I</strong>ng model<strong>S</strong> 🚀
</div>
</div>
<div>
<br>

<div align="center">

[![Notion](https://img.shields.io/badge/Notion-%23000000.svg?style=for-the-badge&logo=notion&logoColor=white)](https://honorable-payment-890.notion.site/POLARIS-A-POst-training-recipe-for-scaling-reinforcement-Learning-on-Advanced-ReasonIng-modelS-1dfa954ff7c38094923ec7772bf447a1)
[![Twitter](https://img.shields.io/badge/twitter-white?style=for-the-badge&logo=X&logoColor=000&color=000&labelColor=white)](https://x.com/AnChancy46881/status/1936099024769368506)
[![Hugging Face Model](https://img.shields.io/badge/models-%23000000?style=for-the-badge&logo=huggingface&logoColor=000&logoColor=white)](https://huggingface.co/POLARIS-Project/Polaris-4B-Preview)
[![Hugging Face Data](https://img.shields.io/badge/data-%23000000?style=for-the-badge&logo=huggingface&logoColor=000&logoColor=white)](https://huggingface.co/datasets/POLARIS-Project/Polaris-Dataset-53K)
[![Paper](https://img.shields.io/badge/Paper-%23000000?style=for-the-badge&logo=arxiv&logoColor=000&labelColor=white)]()
</div>
</div>


## Overview

Polaris is an open-source post-training recipe that leverages reinforcement learning (RL) scaling to further optimize models with strong reasoning capabilities. Our work demonstrates that even state-of-the-art models like Qwen3-4B can achieve remarkable gains on complex reasoning tasks when enhanced with Polaris.
By training with open-source data and academic-grade resources, Polaris elevates the performance of open-recipe reasoning models to an entirely new level. In benchmark evaluations, our approach astonishingly outperforms leading commercial systems such as Claude-4-Opus, Grok-3-Beta, and o3-mini-high(2025/01/03).

This work is done as part of the [HKU NLP Group](https://hkunlp.github.io/) and [Bytedance Seed](https://seed.bytedance.com/). Our training and evaluation codebase is built on [Verl](https://github.com/volcengine/verl). To foster progress in scaling RL on advanced reasoning models, we are open‐sourcing our complete dataset, code, and training details for the research community.


<div align="center">
<img src="figs/aime25.png" width="80%" />
</div>


## 🔥Releases

<strong>[2025/06/20]</strong>
- 🧾 The Blog that details our training recipe: [Notion](https://honorable-payment-890.notion.site/POLARIS-A-POst-training-recipe-for-scaling-reinforcement-Learning-on-Advanced-ReasonIng-modelS-1dfa954ff7c38094923ec7772bf447a1) and [Blog](https://hkunlp.github.io/blog/2025/Polaris)
- 🤗 Model weights: [Polaris-4B-Preview](https://huggingface.co/POLARIS-HKU/Polaris-4B-Preview) and  [Polaris-7B-Preview](https://huggingface.co/POLARIS-Project/Polaris-7B-Preview). Polaris-4B-Preview is fine-tuned from Qwen3-4B and Polaris-7B-Preview is fine-tuned from Deepseek-R1-Distill-Qwen-7B.
- 📚 The filtered training dataset with difficulty distribution  [Polaris-Dataset-53K](https://huggingface.co/datasets/POLARIS-Project/Polaris-Dataset-53K)
- ⏰ Full training code and training scripts will be available in one week. 

## Running environment 
```bash
cd Polaris
pip install -e ./verl 
pip install -e ./
pip install transformers==4.51.0
pip install vllm==0.8.4
pip install tensordict==0.6.2

# do not use xformers backend
unset VLLM_ATTENTION_BACKEND

```
## Demo
```python
import torch
from transformers import AutoTokenizer
from vllm import SamplingParams, LLM


example = {
        "question": "Find the largest possible real part of \\[(75+117i)z+\\frac{96+144i}{z}\\]where $z$ is a complex number with $|z|=4$.\nLet's think step by step and output the final answer within \\boxed{}.",
        "answer": "540"
}


model = "/path/to/Polaris-4B-Preview"

tokenizer = AutoTokenizer.from_pretrained(model)

llm = LLM(
    model=model,
    dtype=torch.bfloat16,
    tensor_parallel_size=1,
    gpu_memory_utilization=0.9
)

sampling_params = SamplingParams(
    temperature=1.4,
    top_p=1.0,
    max_tokens=90000
)

question = example["question"]
answer = example["answer"]
output = llm.generate(
            prompts=tokenzier.apply_chat_template(conversation=[{"content": question, "role": "user"}],
                                                  add_generation_prompt=True,
                                                  tokenize=False),
            sampling_params=sampling_params
        )
print(f"***QUESTION***:\n{question}\n***GROUND TRUTH***:\n{answer}\n***MODEL OUTPUT***:\n{output[0].outputs[0].text}\n")
```


## Training
### Step1: Data preparation
The [training data](https://huggingface.co/datasets/POLARIS-Project/Polaris-Dataset-53K) used in this work is filtered from [DeepScaleR-dataset-40K](https://huggingface.co/datasets/agentica-org/DeepScaleR-Preview-Dataset) and [AReaL-dataset-106K](https://huggingface.co/datasets/inclusionAI/AReaL-boba-Data).
We provide the [parquet data]() which can be directly used in training. 
If your data is in `json` or `jsonl` format, please use the following cmd for converting it into the parquet format. 
```bash
# Generate parquet files for parquet_data/{jsonl_file_name}.parquet 
python scripts/data/polaris_dataset.py --jsonl_file data/jsonl_data/polaris-data-53K.jsonl # => data/jsonl_data/polaris-data-53K.parquet
```
### Step2: Temperature searching for diversity rollouts
Temperature searching is highly recommended before each stage of training as suggested by the `diversity-based rollouts sampling` section in our blog.
```bash
# the following code will provide the optimal training temperature for you
cd evaluation
python search_optimal_temperature.py --start 0.6 (recommended decoding temperature by the model developers) --end 1.5 --step 0.1 --model /path/to/qwen3-4b (base model) --n 16 --new_tokens 50000
# after the searching process, run:
python get_optimal_temperature.py  --start 0.6 --end 1.5 --step 0.1 --model /path/to/qwen3-4b (base model)
```

### Step3: Multi-stage Training
The training scripts for Polaris that details are available [here]()
The training process for Polaris-4B-preview requires at least 4 nodes. 
Our multi-node training is based on Ray. Please run the command on **all nodes**.

#### Stage1-training 
```bash
# run ray stop if needed
python train_with_ray.py  --model /path/to/qwen3-4b --name Polaris-4B-stage1 (your experiment name) --n_nodes 4  --head True/False (True for head node)  --sh ./scripts/train/polaris_4b_stage1_40k_t1.4.sh
```

#### Stage2-training
```bash
# convert the checkpoint after stage1-training to hf model
python verl/scripts/model_merger.py --local_dir /path/to/checkpoints/global_step_XXX/actor --target_dir /path/to/hf/stage1-checkpoint
# run ray stop if needed
python train_with_ray.py  --model /path/to/hf/stage1-checkpoint --name Polaris-4B-stage2 --n_nodes 4  --head True/False   --sh ./scripts/train/polaris_4b_stage2_48k_t1.45.sh
```

#### Stage3-training 
```bash
# convert the checkpoint after stage1-training to hf model
python verl/scripts/model_merger.py --local_dir /path/to/checkpoints/global_step_XXX/actor --target_dir /path/to/hf/stage2-checkpoint
# run ray stop if needed
python train_with_ray.py  --model /path/to/hf/stage2-checkpoint --name Polaris-4B-stage3 --n_nodes 4  --head True/False  --sh ./scripts/train/polaris_4b_stage3_52k_t1.5.sh
```


## 📊Evaluation
We recommend using a higher temperature for decoding than that suggested for Qwen3 (0.6 → 1.4). However, it is not advisable to exceed the temperature used during training. For POLARIS, a longer response length (> 64K) should be utilized to prevent performance degradation from truncation, which could otherwise cause its performance to fall below that of Qwen3. All other settings remain the same. 

**Evaluation command based on verl**:
```bash
./scripts/eval/eval_model_aime24.sh --model [CHECKPOINT_PATH]  --n 32 --max_length 90000  --t 1.4
./scripts/eval/eval_model_aime25.sh --model [CHECKPOINT_PATH]  --n 32 --max_length 90000  --t 1.4 or 1.45
```

Example inference

### Results 

| **Models** | **AIME24 avg@32** | **AIME25 avg@32** | **Minerva Math avg@4** | **Olympiad Bench avg@4** | **AMC23  avg@8** |
| --- | --- | --- | --- | --- | --- |
| `Deepseek-R1-Distill-Qwen-7B` | 55.0 | 39.7 | 36.7 | 56.8 | 81.9 |
| `AReal-boba-RL-7B` | 61.9 | 48.3 | 39.5 | 61.9 | 86.4 |
| `Skywork-OR1-7B-Math` | 69.8 | 52.3 | **40.8** | 63.2 | 85.3 |
| **`POLARIS-7B-Preview`** | **72.6** | **52.6** | 40.2 | **65.4** | **89.0** |
| `Deepseek-R1-Distill-Qwen-32B` | 72.6 | 54.9 | 42.1 | 59.4 | 84.3 |
| `qwen3-32B` | 81.4 | 72.9 | 44.2 | 66.7 | 92.4 |
| `qwen3-4B` | 73.8 | 65.6 | 43.6 | 62.2 | 87.2 |
| **`POLARIS-4B-Preview`** | **81.2** | **79.4** | **44.0** | **69.1** | **94.8** |


## Acknowledgements
The training and evaluation codebase is heavily built on [Verl](https://github.com/volcengine/verl). The reward function in polaris is from [DeepScaleR](https://github.com/agentica-project/rllm). Our model is trained on top of [`Qwen3-4B`](https://huggingface.co/Qwen/Qwen3-4B) and [`DeepSeek-R1-Distill-Qwen-7B`](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B). Thanks for their wonderful work.


## Citation
```bibtex
@misc{Polaris2025,
    title = {POLARIS: A Post-Training Recipe for Scaling Reinforcement Learning on Advanced Reasoning Models},
    url = {https://hkunlp.github.io/blog/2025/Polaris},
    author = {An, Chenxin and Xie, Zhihui and Li, Xiaonan and Li, Lei and Zhang, Jun and Gong, Shansan and Zhong, Ming and Xu, Jingjing and Qiu, Xipeng and Wang, Mingxuan and Kong, Lingpeng}
    year = {2025}
}
```

