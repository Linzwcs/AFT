# Aggregation Fine-Tuning for Large Language Models

<div align="center">
  <a href="https://github.com/OpenSparseLLMs/Skip-DiT"><img src="https://img.shields.io/static/v1?label=Skip-DiT-Code&message=Github&color=blue&logo=github-pages"></a> &ensp;
  <a href="https://arxiv.org/abs/2411.17616"><img src="https://img.shields.io/static/v1?label=Paper&message=Arxiv:Skip-DiT&color=red&logo=arxiv"></a> &ensp;
  <a href="https://huggingface.co/GuanjieChen/Skip-DiT"><img src="https://img.shields.io/static/v1?label=Skip-DiT&message=HuggingFace&color=yellow"></a> &ensp;
</div>

<div align="center">
  <img src="./figs/methods.jpg" width="85%" ></img>
  <br>
  <em>
      (Framework of aggregation fine-tuning and propose-and-aggregate inference.) 
  </em>
</div>
<br>

### 🎉🎉🎉 About
This repository contains the official PyTorch implementation of the paper: **[From Drafts to Answers: Aggregation Fine-Tuning for Large Language Models](https://arxiv.org/abs/2411.17616)**. 
In this work, we introduce Aggregation Fine-Tuning (**AFT**), a supervised fine-tuning paradigm where the model learns to synthesize multiple draft responses, referred to as prposals, into a single, refined answer, termed aggregation. 
An AFT model, fine-tuned from Llama3.1-8B-Base with only 64K data, achieves a 41.3\% LC win rate on AlpacaEval 2, surpassing significantly larger LLMs such as Llama3.1-405B-Instruct and GPT-4.
Our analysis reveals that aggregation learning converges faster and more stably by operating within a low-perplexity region shaped by the aggregation data.
All the codes and checkpoints are publicly available at [huggingface]() and [github](). 


### 🔥 News
(🔥News) Dec 12, 2024🔥 inference code of [**AFT**](https://huggingface.co/GuanjieChen/Skip-DiT/tree/main) is now fully released 🎉.


### 🔍 Install





### 🚀🚀🚀 Quick Start


####  Inference




### 🏋️🏊🏃 Training


### 🛒 Pretrained Models


### 🌺🌺🌺 Acknowledgement


### License
The code and model weights are licensed under [LICENSE](./LICENSE).

