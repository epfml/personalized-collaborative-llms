# Personalized collaborative fine-tuning for LLMs

This is the code base for the paper [Personalized Collaborative Fine-Tuning for On-Device Large Language Models](https://arxiv.org/abs/2404.09753)

## Quickstart

Install conda environment:

```
conda env create -f env.yml && conda activate llm-lora
```

#### Generate dataset manually

If you want to generate datasets before running fine-tuning:

```
python ./src/gen_dataset.py <dataset_name>
```

### Base command

This is base command to run suitable experiments for Nvidia A100 GPU:

```
python ./src/main.py --config_format lora --use_pretrained gpt2 \
--eval_freq 25 --pretraining_rounds 100 --iterations 500 \
--lora_mlp --lora_causal_self_attention --lora_freeze_all_non_lora \
--trust <trust> --dataset=<dataset_name> --num_clients <num_clients>
```

### Scripts

To reproduce some of the experiments you can run the following command (this might take a while to run):

```
./scripts/script.sh <wandb_project> <dataset_name> <trust> <num_clients>
```



