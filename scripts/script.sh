#!/bin/bash

if [ "$#" -ne 4 ]; then
    echo "Usage: $0 wandb_project dataset_name trust num_clients"
    exit 1
fi

wandb_project=$1
dataset_name=$2
trust=$3
num_clients=$4

for seed in {1..10}; do
    echo "---------------- New sample with seed: $seed ----------------"

    python -W ignore ./src/main.py --seed $seed --trust $trust --wandb_project "$wandb_project" --wandb_group "$trust" --trust_freq 25 --pretraining_rounds 100 --iterations 500 --num_clients $num_clients --eval_freq 25 --dataset $dataset_name --wandb --config_format lora --use_pretrained gpt2 --lora_mlp --lora_causal_self_attention --lora_freeze_all_non_lora
done



