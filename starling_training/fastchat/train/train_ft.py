# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import os
import sys
from dataclasses import dataclass, field
import logging
import pathlib
import typing
import torch

from deepspeed import zero
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, dir_path + "/../..")

from transformers import LlamaForCausalLM, LlamaTokenizer, AutoModelForCausalLM
import transformers
from transformers import Trainer

from fastchat.train.train import (
    DataArguments,
    ModelArguments,
    TrainingArguments,
    make_supervised_data_module,
)


def train():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    (
        model_args,
        data_args,
        training_args
    ) = parser.parse_args_into_dataclasses()
    
    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        
    print("device map:", device_map)

    model = LlamaForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        load_in_8bit=False,
        torch_dtype=torch.float16,
        device_map=device_map
    )

    def make_inputs_require_grad(module, input, output):
        output.requires_grad_(True)
        
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    tokenizer.pad_token = tokenizer.unk_token
    
    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    
    if torch.cuda.device_count() > 1:
        model.is_parallelizable = True
        model.model_parallel = True
    model.config.use_cache = False
    
    print(f"\n\n#Number trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    trainer = Trainer(
        model=model, tokenizer=tokenizer, args=training_args, **data_module
    )

    #print(f"\n\n Device Map: {model.hf_device_map}")
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
        
    trainer.save_state()
    
    model.save_pretrained(training_args.output_dir)

if __name__ == "__main__":
    with torch.autocast("cuda"):
        train()
