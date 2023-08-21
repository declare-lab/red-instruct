# Red-Teaming Large Language Models using Chain of Utterances for Safety-Alignment

[**Paper**](https://openreview.net/pdf?id=jkcHYEfPv3) | [**Github**](https://github.com/declare-lab/red-instruct) | [**Dataset**](https://huggingface.co/datasets/declare-lab/HarmfulQA) | [**Model**](https://huggingface.co/declare-lab/starling-7B)

📢 Training codes will be released soon. Stay tuned!

**As a part of our efforts to make LLMs safer for public use, we provide:**
- **Code to evaluate LLM safety against Chain of Utterances (CoU) based prompts-referred to as RedEval benchmark**
- **Code to perform safety alignment of Vicuna-7B on blue-red data from [**HarmfulQA**](https://huggingface.co/datasets/declare-lab/HarmfulQA)**

## Red-Eval Benchmark
Simple scripts to evaluate closed-source systems (ChatGPT, GPT4) and open-source LLMs on our benchmark red-eval.

To compute Attack Success Rate (ASR) Red-Eval uses two question-bank consisting of harmful questions:
- [**HarmfulQA**](https://huggingface.co/datasets/declare-lab/HarmfulQA) (1,960 harmful questions covering 10 topics and ~10 subtopics each)
- [**DangerousQA**](https://github.com/SALT-NLP/chain-of-thought-bias/blob/main/data/dangerous-q/toxic_outs.json) (200 harmful questions across 6 adjectives—racist, stereotypical, sexist, illegal, toxic, and harmful) 

### How to perform red-teaming
- Step-1: Generate model outputs on harmful questions by providing a path to the question bank and red-teaming prompt.

Closed-source models (GPT4 and ChatGPT):
```
  python generate_responses.py --model 'gpt4' --prompt 'red_prompts/cou.txt' --dataset hamrful_questions/dangerousqa.json
  python generate_responses.py --model 'chatgpt' --prompt 'red_prompts/cou.txt' --dataset hamrful_questions/dangerousqa.json
```

  Open-source models:
  
```
  python generate_responses.py --model lmsys/vicuna-7b-v1.3 --prompt 'red_prompts/cou.txt' --dataset hamrful_questions/dangerousqa.json
```
  For better readability, internal thoughts from responses can be cleaned by specifying "clean_thoughts" in the script:
```
python generate_responses.py --model 'gpt4' --prompt 'red_prompts/cou.txt' --dataset hamrful_questions/dangerousqa.json --clean_thoughts
python generate_responses.py --model 'chatgpt' --prompt 'red_prompts/cou.txt' --dataset hamrful_questions/dangerousqa.json --clean_thoughts
python generate_responses.py --model lmsys/vicuna-7b-v1.3 --prompt 'red_prompts/cou.txt' --dataset hamrful_questions/dangerousqa.json --clean_thoughts
```

- Step-2: Annotate the generated responses using gpt4-as-a-judge
```
python gpt4_as_judge.py --response_file "results/dangerousqa_gpt4_cou_clean.json" --save_path results
```
