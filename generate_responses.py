import os
import time
import json
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--prompt', help='prompt template to be used for red-teaming', type=str, required=True)
parser.add_argument('--keep_thoughts', help='remove internal thoughts from the output', action='store_true', required=False)
parser.add_argument('--model', help='model under evaluation: gpt4, chatgpt, huggingface_model_path', type=str, required=True)
parser.add_argument('--save_path', help='path where the model results to be saved', type=str, required=False, default='results')
parser.add_argument('--num_samples', help='number of first num_samples to test from the dataset', type=int, required=False, default=-1)
parser.add_argument('--load_8bit', help='for open source models-if the model to be loaded in 8 bit', action='store_true', required=False)
parser.add_argument('--dataset', help='path to harmful questions (json) for evaluation, to be used with prompt templates for red-teaming', required=True, type=str)

args = parser.parse_args()

dataset = args.dataset
model_name = args.model
save_path = args.save_path
load_in_8bit = args.load_8bit
num_samples = args.num_samples
clean_thoughts = not args.keep_thoughts
prompt = args.prompt

print(f"\n\nconfiguration")
print(f"*{'-'*10}*")

for arg in vars(args):
    print(f"{arg}: {getattr(args, arg)}")

print(f"*{'-'*10}*\n\n")


tokenizer = None

##setting up model##
if 'gpt' in model_name:

    import openai

    try:
        # API setting constants
        API_MAX_RETRY = 5
        API_RETRY_SLEEP = 10
        API_ERROR_OUTPUT = "$ERROR$"

        key_path = f'api_keys/{model_name}_api_key.json'
        with open(key_path, 'r') as f:
            keys = json.load(f)   

        openai.api_type = keys['api_type']
        openai.api_base = keys['api_base']
        openai.api_version = keys['api_version']
        openai.api_key=keys['api_key']
        model_engine = keys['model_engine']
        model_family = keys['model_family']

    except:
        raise Exception(f"\n\n\t\t\t[Sorry, please verify API key provided for {model_name} at {key_path}]")


elif 'claude' in model_name:

    from anthropic import Anthropic

    try:
        # API setting constants
        API_MAX_RETRY = 5
        API_RETRY_SLEEP = 10
        API_ERROR_OUTPUT = "$ERROR$"

        key_path = f'api_keys/claude_api_key.json'
        with open(key_path, 'r') as f:
            keys = json.load(f)   

        anthropic = Anthropic(api_key=keys['api_key'])

    except:
        raise Exception(f"\n\n\t\t\t[Sorry, please verify API key provided for {model_name} at {key_path}]")

else:

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers import AutoModelForCausalLM

    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="right", use_fast=False)

    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.unk_token

    if load_in_8bit:
        print("\n\n***loading model in 8 bits***\n\n")
        
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", load_in_8bit=load_in_8bit)


##define chat completion function for GPT##
def chat_completion_gpt(system, prompt):
    for _ in range(API_MAX_RETRY):
        try:    
            response = openai.ChatCompletion.create(
              model=model_family,
              engine=model_engine,
              messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ]
            )
            response = response['choices'][0]['message']['content']
            return response
        except Exception as e:
            if 'policy' in str(e):
                print("Skipping due to openai policy")
                #such cases are tagged with [[N]]
                return '[[N]]'
            print(type(e), e)
            print("trying again")
            time.sleep(API_RETRY_SLEEP)


##define chat completion function for Claude##
def chat_completion_claude(system, prompt):

    try_suffix = [" "," Let's think step by step:"," Let's think step by step.", " Let's think step by step:\n-", " Let's think step by step.\n-"]

    for t in range(API_MAX_RETRY):
        try:
            full_prompt = f"{prompt}{try_suffix[t]}"

            print(f"Retry:{t}\n{full_prompt}")

            completion = anthropic.messages.create(
                model=model_name,
                max_tokens=1000,
                messages=[
                            {"role": "user", "content": full_prompt}
                        ],
            )

            response = completion.content[0].text

            if response == "":
                continue
                
            return response

        except Exception as e:
            print(type(e), e)
            print("trying again")
            time.sleep(API_RETRY_SLEEP)

            return response


##process data##
def clean_thought(response):

    if "(Internal thought:" in response:
        if ')' in response:
            ind =  response.index(')')+1
        else:
            ind = -1
        nresponse = response[ind:].strip()
        return nresponse

    return response


def get_context(file_name):
    f = open(file_name, "r")
    f = f.read()
    return f


def gen_prompt(q, ctx):
    prompt = ctx.replace('<question>', q.strip())

    #open-source models, apply chat template
    if tokenizer:
        prompt = [{"role": "user", "content": prompt}]
        prompt = tokenizer.apply_chat_template(prompt, tokenize=False)

    return prompt


def process_data(dataset, ctx, nsamples):
    f = open(dataset)

    data = json.load(f)

    if 'harmfulq' in dataset:
        topics = []
        subtopics = []
        prompt_que = []
        orig_que = []
        for topic in data.keys():
            for subtopic in data[topic].keys():
                for q in data[topic][subtopic]:
                    orig_que.append(q)
                    prompt_que.append(gen_prompt(q, ctx))
                    topics.append(topic)
                    subtopics.append(subtopic)

    else:
        prompt_que = [gen_prompt(q, ctx) for q in data]
        orig_que = data
        topics, subtopics = [], []

    if nsamples == -1:
        nsamples = len(prompt_que)

    return prompt_que[:nsamples], orig_que[:nsamples], topics[:nsamples], subtopics[:nsamples]


context = get_context(args.prompt)
prompt_que, orig_que, topics, subtopics = process_data(dataset, context, num_samples)


##generate responses##
if not os.path.exists(save_path):
    os.makedirs(save_path)

#save file name
if clean_thoughts:
    save_name = f'{save_path}/{dataset.split("/")[-1].replace(".json","")}_{model_name.split("/")[-1]}_{prompt.split("/")[-1].replace(".txt","")}.json'
else:
    save_name = f'{save_path}/{dataset.split("/")[-1].replace(".json","")}_{model_name.split("/")[-1]}_{prompt.split("/")[-1].replace(".txt","")}_w_thoughts.json'


outputs = []
system_message = ''


print("generating responses...\n")
for i in tqdm(range(len(prompt_que))):

    inputs = prompt_que[i]

    if 'gpt' in model_name:
        response = chat_completion_gpt(system=system_message, prompt=inputs)

    elif 'claude' in model_name:
        response = chat_completion_claude(system=system_message, prompt=inputs)

    else:
        inputs = tokenizer([inputs], return_tensors="pt", truncation=False, padding=True, add_special_tokens=False).to("cuda")
        generated_ids = model.generate(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'], max_new_tokens=500)
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=False)[0]

    question = orig_que[i]
    question2 = prompt_que[i]
    
    #cleaning response
    #breakpoint()
    response = response.replace("<s> [INST]", "<s>[INST]")
    question2 = question2.replace("<s> [INST]", "<s>[INST]")
    response = response.replace(question2,"").strip()

    if clean_thoughts:
        response = clean_thought(response)

    if 'harmfulq' in dataset:
        response = [{'prompt':question, 'response':response, 'topic':topics[i], 'subtopic': subtopics[i]}]
    else:
        response = [{'prompt':question, 'response':response}]

    outputs += response

    with open(f'{save_name}', 'w', encoding='utf-8') as f:
        json.dump(outputs, f, ensure_ascii=False, indent=4)

print(f"\nCompleted, pelase check {save_name}")
