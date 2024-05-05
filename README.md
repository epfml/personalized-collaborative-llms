# Personalized collaborative fine-tuning for LLMs

This is the code base for the
paper [Personalized Collaborative Fine-Tuning for On-Device Large Language Models](https://arxiv.org/abs/2404.09753)

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

### Reproducibility

To reproduce the experiments you can run the following commands. There are several point to take note of:

1. This might take a while to run, because we run the experiments with 10 different seeds.
2. There two variants to the datasets.
3. For clearer visualisation on wandb, we used a different project for each dataset.
4. The number of clients used can be changed but there is a specific range for each dataset.

#### AG News

News articles in four categories: “World”, “Sports”, “Business” and “Sci/Tech”.

```
./scripts/script.sh <wandb_project> <dataset_name> <trust> <num_clients>
```

#### Multilingual Wikipedia

Wikipedia texts in three languages (categories): French, Italian, and German.

```
./scripts/script.sh <wandb_project> <dataset_name> <trust> <num_clients>
```

#### Codes-Wikipedia (Eng)

The first category is Java code from GitHub (HuggingFace) and the second category is English Wikipedia text

```
./scripts/script.sh <wandb_project> <dataset_name> <trust> <num_clients>
```

## Code structure

```
src                              # Main source folder                                                                                                                                                                                                                                                               
├── config                         # Config files for differents models
│   ├── __init__.py                # Chooses the correct config
│   └── lora.py                    # Configuration parameters for LoRA models
├── data                          # Datasets folder
│   ├── agnews.py                  # AGNews dataset
│   ├── fed_cc_news.py             # Comon Crawl News Federated Leraning dataset 
│   ├── github_wiki.py             # Java github code and wikipedia EN dataset
│   ├── three_multi.py             # Wikipedia FR, DE and IT
│   ├── utils.py                   # Get the correct dataset from the configuration parameters
│   ├── wikitext.py                # EN Wikipedia, single client test
│   └── wikitext_split.py          # EN, FR, IT or DE, single client test
├── distributed                   # Distributed package to run experiments on multiple GPU, NOT used in our experiments
│   └── ...                        # Only the default, single backend is used
├── gen_dataset.py                 # Allows to generate only datasets
├── main.py                        # Put everything togheter to fine-tune LoRA model on the various dataset
├── models                        # Models definition
│   ├── lora.py                    # LoRA nano-GPT model definition
│   └── utils.py                   # Get the correct model from the configuration parameters
└── optim                         # Training/Fine-tuning loop for the models 
    ├── lora.py                    # Loop for LoRA model
    └── utils.py                   # General useful methods
```