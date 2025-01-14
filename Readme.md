# Aggregation Fine-Tuning for Large Language Models

<div align="center">
  <a href="https://github.com/Linzwcs/AFT/tree/main"><img src="https://img.shields.io/static/v1?label=AFT&message=Github&color=blue&logo=github-pages"></a> &ensp;
  <a href=""><img src="https://img.shields.io/static/v1?label=Paper&message=Arxiv:AFT&color=red&logo=arxiv"></a> &ensp;
  <a href="https://huggingface.co/GuanjieChen/Skip-DiT"><img src="https://img.shields.io/static/v1?label=AFT&message=HuggingFace&color=yellow"></a> &ensp;
</div>

<div align="center">
  <img src="./figs/methods.jpg" width="85%" ></img>
  <br>
  <em>
      (Framework of aggregation fine-tuning and propose-and-aggregate inference.) 
  </em>
</div>
<br>

## 🎉🎉🎉 About
This repository contains the official PyTorch implementation of the paper: **[From Drafts to Answers: Aggregation Fine-Tuning for Large Language Models](https://arxiv.org/abs/2411.17616)**. 
In this work, we introduce Aggregation Fine-Tuning (**AFT**), a supervised fine-tuning paradigm where the model learns to synthesize multiple draft responses, referred to as prposals, into a single, refined answer, termed aggregation. 
An AFT model, fine-tuned from Llama3.1-8B-Base with only 64K data, achieves a 41.3\% LC win rate on AlpacaEval 2, surpassing significantly larger LLMs such as Llama3.1-405B-Instruct and GPT-4.
Our analysis reveals that aggregation learning converges faster and more stably by operating within a low-perplexity region shaped by the aggregation data.
All the codes and checkpoints are publicly available at [huggingface]() and [github](). 


### 🔥 News
(🔥News) Jan 1, 2025🔥The inference code sample for [**AFT**](https://github.com/Linzwcs/AFT/tree/main) has been released 🎉.


## 🚀🚀🚀 Quick Start


### 🔍 Install


To install the inference framework, follow the steps below:

1. **Create and activate a new environment:**
   ```bash
   conda create -n AFT python=3.11
   conda activate AFT
   ```

2. **Install PyTorch based on your device configuration:**

   Our device uses CUDA 12.1, so install PyTorch with the following command:
   ```bash
   pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 
   ```

3. **Clone the repository, navigate to the directory, and install our inference framework:**
   ```bash
   git clone git@github.com:Linzwcs/AFT.git
   cd AFT
   pip install -e .
   ```

### 🔍  Inference

We provide a sample inference code in inference.py. You can execute it by running the following command:

```bash
  python inference.py \
        --config ./configs/default.yaml \
        --input_file ./data/sample.jsonl \
        --output_file ./output/output.jsonl \
        --batch_size 16
```

The detailed meanings of the keys in the config file are illustrated below：

```yaml
  model_name: <path to AFT model>
  
  # The proposal_params and aggregation_params can be seen to as SamplingParams 
  # and will be sent to vllm.chat().
  proposal_params:
      temperature: 0.7
      top_p: 0.95
      max_tokens: 4096
      n: 5

  aggregation_params:
      temperature: 0.7
      top_p: 0.95
      max_tokens: 4096
      n: 5 

  vllm_seed: 2024 # vllm backend seed

  hidden_layer: 0  # Number of aggregation layers, excluding the final layer
  final_aggregation: True # Indicates whether this is the final aggregation step
```


### 🛒 Released Models



Currently, we only release the `Llama-AFT-On-Policy` model, and you can set `model_name` to this model.

**Supported Models**:

1. `Llama-AFT-On-Policy`


## Subjective Evaluation
We performed evaluations on the MT-Bench and AlpacaEval 2, and the results are presented below:

<table border="1">
  <thead>
    <tr>
      <th><strong>Model</strong></th>
      <th><strong>MT-Bench 1st turn</strong></th>
      <th><strong>MT-Bench 2nd turn</strong></th>
      <th><strong>MT-Bench Avg.</strong></th>
      <th><strong>AlpacaEval 2 LC(%)</strong></th>
      <th><strong>AlpacaEval 2 WR(%)</strong></th>
    </tr>
  </thead>
  <tbody>
    <tr>
     <td colspan="6" style="text-align: center;"><strong>Mistral-7B-v0.1-Base</strong></td>
  </tr>
    <tr>
      <td>SFT</td>
      <td>6.6</td>
      <td>6.1</td>
      <td>6.4</td>
      <td>6.7</td>
      <td>6.1</td>
    </tr>
    <tr>
      <td>AFT-off-policy</td>
      <td>7.7</td>
      <td>6.3</td>
      <td>7.0</td>
      <td>19.8</td>
      <td>20.0</td>
    </tr>
    <tr>
      <td>w/ Agg.</td>
      <td>8.0</td>
      <td>7.0</td>
      <td>7.5</td>
      <td><strong>33.8</strong></td>
      <td>47.8</td>
    </tr>
    <tr>
      <td>AFT-on-policy</td>
      <td>7.5</td>
      <td>6.4</td>
      <td>6.9</td>
      <td>23.4</td>
      <td>24.9</td>
    </tr>
    <tr>
      <td>w/ Agg.</td>
      <td><strong>8.3</strong></td>
      <td><strong>7.0</strong></td>
      <td><strong>7.6</strong></td>
      <td>30.7</td>
      <td><strong>48.4</strong></td>
    </tr>
    <tr>
      <td colspan="6" style="text-align: center;"><strong>Llama3.1-8B-Base</strong></td>
   </tr>
    <tr>
      <td>SFT</td>
      <td>7.3</td>
      <td>6.2</td>
      <td>6.8</td>
      <td>8.0</td>
      <td>7.3</td>
    </tr>
    <tr>
      <td>AFT-off-policy</td>
      <td>7.7</td>
      <td>6.9</td>
      <td>7.3</td>
      <td>20.3</td>
      <td>19.6</td>
    </tr>
    <tr>
      <td>w/ Agg.</td>
      <td>8.3</td>
      <td><strong>7.6</strong></td>
      <td>7.9</td>
      <td>40.3</td>
      <td>47.8</td>
    </tr>
    <tr>
      <td>AFT-on-policy</td>
      <td>7.9</td>
      <td>6.9</td>
      <td>7.4</td>
      <td>21.5</td>
      <td>21.8</td>
    </tr>
    <tr>
      <td>w/ Agg.</td>
      <td><strong>8.5</strong></td>
      <td><strong>7.6</strong></td>
      <td><strong>8.1</strong></td>
      <td><strong>41.3</strong></td>
      <td><strong>51.3</strong></td>
    </tr>
  </tbody>
</table>


      
### 🌺🌺🌺 Acknowledgement


### License
The code and model weights are licensed under [LICENSE](./LICENSE).

