#!/bin/bash

if [ "$#" -ne 5 ]; then
    echo "Usage: $0 wandb_project dataset_name trust num_clients seed"
    exit 1
fi

wandb_project=$1
dataset_name=$2
trust=$3
num_clients=$4
seed_lim=$5

for ((seed = 1; seed <= seed_lim; seed++)); do
    echo "---------------- New sample with seed: $seed ----------------"

    python -W ignore ./src/main.py --seed $seed --trust $trust --wandb_project "$wandb_project" --wandb_group "$trust" --trust_freq 25 --pretraining_rounds 26 --iterations 5000 --num_clients $num_clients --eval_freq 25 --dataset $dataset_name --wandb --config_format lora --use_pretrained gpt2 --lora_causal_self_attention --lora_freeze_all_non_lora
done



