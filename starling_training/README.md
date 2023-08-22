# Training script to obtain Starling (Fine-tune Vicuna-7B on Blue data)

## Installation
```
conda create --name LLMFT -c conda-forge python=3.11
conda activate LLMFT
conda install pip
pip install torch torchvision torchaudio
pip install -e .
pip install deepspeed
pip install scikit-learn
```

## Process data

## Train
- Process data by combining blue + helpful + equal amount of ShareGPT
```
python process_data.py
```

- Tune  Vicuna-7B
```
CUDA_VISIBLE_DEVICES=0,1,2,3 deepspeed --num_gpus=4 fastchat/train/train_ft.py \
    --ddp_timeout=360000 \
    --output_dir "./saved/starling/" \
    --deepspeed "deepspeed_configs/fp16_ft_7b.json" --fp16 True \
    --model_name_or_path "lmsys/vicuna-7b-v1.3" \
    --data_path "./data/data_combined.json" \
    --per_device_train_batch_size 2 --per_device_eval_batch_size 4 --gradient_accumulation_steps 8 \
    --num_train_epochs 3 --evaluation_strategy "epoch" \
    --save_strategy "steps" --save_steps 200 --save_total_limit 20 \
    --learning_rate 1e-5 --weight_decay 0. --warmup_ratio 0.0001 --lr_scheduler_type "cosine" --logging_steps 10 \
    --model_max_length 1280 \
    --gradient_checkpointing True --lazy_preprocess True --disable_tqdm False
```

_Note: We adopt [FastChat](https://github.com/lm-sys/FastChat) to train Starling on the Blue data of HarmfulQA._

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
```
