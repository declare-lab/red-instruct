# Red-Teaming Large Language Models using Chain of Utterances for Safety-Alignment

[**Paper**](https://arxiv.org/abs/2308.09662) | [**Github**](https://github.com/declare-lab/red-instruct) | [**Dataset**](https://huggingface.co/datasets/declare-lab/HarmfulQA) | [**Model**](https://huggingface.co/declare-lab/starling-7B)


> 📣 Update 2/02/24: Introducing Resta: **Safety Re-alignment of Language Models**. [**Paper**](https://arxiv.org/abs/2402.11746) [**Github**](https://github.com/declare-lab/resta) [**Dataset**](https://huggingface.co/datasets/declare-lab/CategoricalHarmfulQA)

> 📣 Update 26/10/23: Introducing our new red-teaming efforts: **Language Model Unalignment**. [**Link**](https://arxiv.org/pdf/2310.14303.pdf)

**As a part of our efforts to make LLMs safer for public use, we provide:**
- **Code to evaluate LLM safety against Chain of Utterances (CoU) based prompts-referred to as RedEval benchmark** <img src="https://github.com/declare-lab/red-instruct/assets/32847115/5678d7d7-5a0c-4d07-b600-1029aa58dbdc" alt="Image" width="30" height="30">

- **Code to perform safety alignment of Vicuna-7B on [**HarmfulQA**](https://huggingface.co/datasets/declare-lab/HarmfulQA), with this we obtain a safer version of Vicuna which is more robust against RedEval. Please check out our [**Starling**](https://huggingface.co/declare-lab/starling-7B)**.

## Red-Eval Benchmark
Simple scripts to evaluate closed-source systems (ChatGPT, GPT4) and open-source LLMs on our benchmark red-eval.

To compute Attack Success Rate (ASR) Red-Eval uses two question-bank consisting of harmful questions:
- [**HarmfulQA**](https://huggingface.co/datasets/declare-lab/HarmfulQA) (1,960 harmful questions covering 10 topics and ~10 subtopics each)
- [**DangerousQA**](https://github.com/SALT-NLP/chain-of-thought-bias/blob/main/data/dangerous-q/toxic_outs.json) (200 harmful questions across 6 adjectives—racist, stereotypical, sexist, illegal, toxic, and harmful)
- [**CategoricalQA**](https://huggingface.co/datasets/declare-lab/CategoricalHarmfulQA) (11 categories of harm, each with 5 sub-categories. Available in English, Chinese, and Vietnamese)
- [**AdversarialQA**](https://github.com/llm-attacks/llm-attacks/blob/main/data/transfer_expriment_behaviors.csv) (a set of 500 instructions to tease out harmful behaviors from the model)

### Installation
```
conda create --name redeval -c conda-forge python=3.11
conda activate redeval
pip install -r requirements.txt
conda install sentencepiece

Store your API keys in api_keys directory! It will be used by LLM as judge (response evaluator) and generate_responses.py for closed-source models.
```

### How to perform red-teaming
- **Step-0: Decide which prompt template you want to use for red-teaming.** As a part of our efforts, we provide a CoU-based prompt that is effective at breaking the safety guardrails of GPT4, ChatGPT, and open-source models.
  - [Chain of Utterances (CoU)](https://github.com/declare-lab/red-instruct/blob/main/red_prompts/cou.txt)
  - [Chain of Thoughts (CoT)](https://github.com/declare-lab/red-instruct/blob/main/red_prompts/cot.txt)
  - [Standard prompt](https://github.com/declare-lab/red-instruct/blob/main/red_prompts/standard.txt)
  - [Suffix prompt](https://github.com/declare-lab/red-instruct/blob/main/red_prompts/suffix.txt)

    (_Note: Different LLMs may require slight variations in the above prompt template to generate meaningful outputs. To create a new template, you can refer to the above template files. Just make sure to have a "\<question\>" string in the prompt which is a placeholder for the harmful question._)
    
- **Step-1: Generate model outputs on harmful questions by providing a path to the question bank and red-teaming prompt:**

  Closed-source models:
```
  #OpenAI
  python generate_responses.py --model "gpt4" --prompt red_prompts/[standard/cou/cot].txt --dataset harmful_questions/dangerousqa.json
  python generate_responses.py --model "chatgpt" --prompt red_prompts/[standard/cou/cot].txt --dataset harmful_questions/dangerousqa.json

  #Claude Models
  python generate_responses.py --model "claude-3-opus-20240229" --prompt red_prompts/[standard/cou/cot].txt --dataset harmful_questions/dangerousqa.json
  python generate_responses.py --model "claude-3-sonnet-20240229" --prompt red_prompts/[standard/cou/cot].txt --dataset harmful_questions/dangerousqa.json
  python generate_responses.py --model "claude-2.1" --prompt red_prompts/[standard/cou/cot].txt --dataset harmful_questions/dangerousqa.json
  python generate_responses.py --model "claude-2.0" --prompt red_prompts/[standard/cou/cot].txt --dataset harmful_questions/dangerousqa.json 
```

  Open-source models:
```
  #Llama-2
  python generate_responses.py --model "meta-llama/Llama-2-7b-chat-hf" --prompt red_prompts/[standard/cou/cot].txt --dataset harmful_questions/dangerousqa.json

  #Mistral
  python generate_responses.py --model "mistralai/Mistral-7B-Instruct-v0.2" --prompt red_prompts/[standard/cou/cot].txt --dataset harmful_questions/dangerousqa.json

  #Vicuna
  python generate_responses.py --model "lmsys/vicuna-7b-v1.3" --prompt red_prompts/[standard/cou/cot].txt --dataset harmful_questions/dangerousqa.json
```


To load models in 8-bit, we can specify --load_8bit as follows

```
  python generate_responses.py --model "meta-llama/Llama-2-7b-chat-hf" --prompt red_prompts/[standard/cou/cot].txt --dataset harmful_questions/dangerousqa.json --load_8bit
```

To run on a subset of the harmful questions, we can specify --num_samples as follows

```
  python generate_responses.py --model "meta-llama/Llama-2-7b-chat-hf" --prompt red_prompts/[standard/cou/cot].txt --dataset harmful_questions/dangerousqa.json --num_samples 10
```


- **Step-2: Annotate the generated responses using gpt4-as-a-judge:**
```
python gpt4_as_judge.py --response_file results/dangerousqa_gpt4_cou.json --save_path results
```

### Results
Attack Success Rate (ASR) of different red-teaming attempts.

|    **Model**    | **DangerousQA (Standard)**   |   **DangerousQA (CoT)**   |  **DangerousQA (RedEval)**  |  **DangerousQA (Average)**  | **HarmfulQA (Standard)**   |   **HarmfulQA (CoT)**   |  **HarmfulQA (RedEval)**  |  **HarmfulQA (Average)**  |
|:--------------:|:------------------:|:------------:|:-----------------:|:------------:|:------------:|:------------:|:-----------------:|:------------:|
|     **GPT-4**     |        0         |       0      |      0.651      |     0.217     |       0        |      0.004     |      0.612      |     0.206     |
|    **ChatGPT**    |        0         |     0.005    |      0.728      |     0.244     |     0.018      |    0.027      |      0.728      |     0.257     |
|  **Vicuna-13B**   |     0.027      |     0.490    |      0.835      |     0.450     |       -        |      -        |       -        |       -       |
|  **Vicuna-7B** |     0.025      |     0.532    |      0.875      |     0.477     |       -        |      -        |       -        |       -       |
| **StableBeluga-13B** |     0.026      |     0.630    |      0.915      |     0.523     |       -        |      -        |       -        |       -       |
| **StableBeluga-7B** |     0.102      |     0.755    |      0.915      |     0.590     |       -        |      -        |       -        |       -       |
|**Vicuna-FT-7B**|     0.095      |     0.465    |      0.860      |     0.473     |       -        |      -        |       -        |       -       |
| **Llama2-FT-7B** |     0.722      |     0.860    |      0.896      |     0.826     |       -        |      -        |       -        |       -       |
|**Starling (Blue)** |     0.015      |     0.485    |      0.765      |     0.421     |       -        |      -        |       -        |       -       |
|**Starling (Blue-Red)** |     0.050      |     0.570    |      0.855      |     0.492     |       -        |      -        |       -        |       -       |
|     **Average**    |     0.116      |     0.479    |      0.830      |     0.471     |     0.010      |    0.016      |     0.67       |     0.232     |


## Citation

```bibtex
@misc{bhardwaj2023redteaming,
      title={Red-Teaming Large Language Models using Chain of Utterances for Safety-Alignment}, 
      author={Rishabh Bhardwaj and Soujanya Poria},
      year={2023},
      eprint={2308.09662},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}

@misc{bhardwaj2024language,
      title={Language Models are Homer Simpson! Safety Re-Alignment of Fine-tuned Language Models through Task Arithmetic}, 
      author={Rishabh Bhardwaj and Do Duc Anh and Soujanya Poria},
      year={2024},
      eprint={2402.11746},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
