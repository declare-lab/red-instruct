# Installation
conda create --name LLMFT -c conda-forge python=3.11

conda activate LLMFT

conda install pip

pip install torch torchvision torchaudio

pip install -e .

pip install deepspeed

pip install scikit-learn

### Download data
gsutil cp gs://hhh_aaai/data_hall_sgpt_combined.json data/

### Dry run (testing on 7B saving at 1 step)
_Before running above, please try saving checkpoint with 7B and 1 step_
```
CUDA_VISIBLE_DEVICES=0,1,2,3 deepspeed --num_gpus=4 fastchat/train/train_ft.py \
    --ddp_timeout=360000 \
    --output_dir "./saved/flacuna/" \
    --deepspeed "deepspeed_configs/fp16_ft_7b.json" --fp16 True \
    --model_name_or_path "lmsys/vicuna-7b-v1.3" \
    --data_path "data/data_hall_sgpt_combined.json" \
    --per_device_train_batch_size 2 --per_device_eval_batch_size 4 --gradient_accumulation_steps 8 \
    --num_train_epochs 3 --evaluation_strategy "epoch" \
    --save_strategy "steps" --save_steps 1 --save_total_limit 20 \
    --learning_rate 2e-5 --weight_decay 0. --warmup_ratio 0.0001 --lr_scheduler_type "cosine" --logging_steps 10 \
    --model_max_length 1280 \
    --gradient_checkpointing True --lazy_preprocess True --disable_tqdm False
```

### Main run (13B)
_If checkpoint shards are being saved, please run the main code_
```
CUDA_VISIBLE_DEVICES=0,1,2,3 deepspeed --num_gpus=4 fastchat/train/train_ft.py \
    --ddp_timeout=360000 \
    --output_dir "./saved/flacuna/" \
    --deepspeed "deepspeed_configs/fp16_ft_13b.json" --fp16 True \
    --model_name_or_path "lmsys/vicuna-13b-v1.3" \
    --data_path "data/data_hall_sgpt_combined.json" \
    --per_device_train_batch_size 2 --per_device_eval_batch_size 4 --gradient_accumulation_steps 8 \
    --num_train_epochs 3 --evaluation_strategy "epoch" \
    --save_strategy "steps" --save_steps 100 --save_total_limit 20 \
    --learning_rate 2e-5 --weight_decay 0. --warmup_ratio 0.0001 --lr_scheduler_type "cosine" --logging_steps 10 \
    --model_max_length 1280 \
    --gradient_checkpointing True --lazy_preprocess True --disable_tqdm False
```
