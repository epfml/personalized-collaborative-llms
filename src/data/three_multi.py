import os
from typing import List, Dict

import numpy as np
import tiktoken
from datasets import load_from_disk, load_dataset

TOKENIZER = tiktoken.get_encoding("gpt2")
MAX_NUM_CLIENTS = 20
NUM_CATEGORIES = 3

MULTI_DATA_PATH = os.path.join(os.path.dirname(__file__), "datasets/three_multi/")
MULTI_MIXED_DATA_PATH = os.path.join(MULTI_DATA_PATH, "mixed")
MULTI_SPECIFIC_DATA_PATH = os.path.join(MULTI_DATA_PATH, "specific")

MAX_SPECIFIC_TOKENS = {"train": 840000, "val": 160000,
                       "ref": 300000}  # if changed too much, won't work properly in the splits
MAX_MIXED_TOKENS = {"train": 630000, "val": 160000}
MIN_MIXED_TOKENS = {"train": 210000, "val": 40000}
MIXED_TOKENS_DIST = [
    [3 / 4, 1 / 4, 0.],
    [0., 3 / 4, 1 / 4],
    [1 / 4, 0., 3 / 4],
]
SPECIFIC_TOKENS_DIST = [
    [1., 0., 0.],
    [0., 1., 0.],
    [0., 0., 1.],
]


def save_ref_data(path: str, val_text_per_class: List[str]) -> None:
    ref_data = []
    for i in range(NUM_CATEGORIES):
        ref_text = " ".join(val_text_per_class[i % NUM_CATEGORIES][:2000])
        raw_tokenized_ref = TOKENIZER.encode_ordinary(ref_text)
        ref_data.append(np.array(raw_tokenized_ref, dtype=np.uint16)[:MAX_SPECIFIC_TOKENS["ref"]])

    ref_tokenized = np.concatenate(ref_data)
    print(f"{ref_tokenized.shape} ref")
    ref_tokenized.tofile(os.path.join(path, "ref.bin"))


def save_raw_token(raw_token: List[int], path: str) -> None:
    tokenized = np.array(raw_token, dtype=np.uint16)
    print(f"{path}: {tokenized.shape}")
    tokenized.tofile(path)


def save_specific_data(train_data: List[str], val_data: List[str], path: str) -> None:
    for i in range(MAX_NUM_CLIENTS):
        train_text = " ".join(train_data[i])
        raw_tokenized_train = TOKENIZER.encode_ordinary(train_text)[:MAX_SPECIFIC_TOKENS["train"]]
        save_raw_token(raw_tokenized_train, os.path.join(path, f"train_{i}.bin"))

        val_text = " ".join(val_data[i])
        raw_tokenized_val = TOKENIZER.encode_ordinary(val_text)[:MAX_SPECIFIC_TOKENS["val"]]
        save_raw_token(raw_tokenized_val, os.path.join(path, f"val_{i}.bin"))


def save_mixed_data(train_data: List[str], val_data: List[str], path: str) -> None:
    for i in range(MAX_NUM_CLIENTS):
        train_text = " ".join(train_data[i])
        val_text = " ".join(val_data[i])
        raw_tokenized_train = TOKENIZER.encode_ordinary(train_text)[:MAX_MIXED_TOKENS["train"]]
        raw_tokenized_val = TOKENIZER.encode_ordinary(val_text)[:MAX_MIXED_TOKENS["val"]]

        train_text = " ".join(train_data[MAX_NUM_CLIENTS + ((i + 1) % MAX_NUM_CLIENTS)])
        val_text = " ".join(val_data[MAX_NUM_CLIENTS + ((i + 1) % MAX_NUM_CLIENTS)])
        raw_tokenized_train.extend(TOKENIZER.encode_ordinary(train_text)[:MIN_MIXED_TOKENS["train"]])
        raw_tokenized_val.extend(TOKENIZER.encode_ordinary(val_text)[:MIN_MIXED_TOKENS["val"]])

        save_raw_token(raw_tokenized_train, os.path.join(path, f"train_{i}.bin"))
        save_raw_token(raw_tokenized_val, os.path.join(path, f"val_{i}.bin"))


def save_train_val_data(dist: str, train_text_per_class: List[str], val_text_per_class: List[str], path: str) -> int:
    match dist:
        case "mixed":
            iters = MAX_NUM_CLIENTS * 2
        case "specific":
            iters = MAX_NUM_CLIENTS
        case _:
            raise NotImplementedError(f"{dist} is not implemented")

    train_data = []
    val_data = []

    end = None
    for i in range(iters):
        start = (i // NUM_CATEGORIES) * 1500  # 800000 tokens
        end = ((i // NUM_CATEGORIES) + 1) * 1500
        train_data.append(train_text_per_class[i % NUM_CATEGORIES][start:end])
        start = (i // NUM_CATEGORIES) * 300  # 160000 tokens
        end = ((i // NUM_CATEGORIES) + 1) * 300
        val_data.append(val_text_per_class[i % NUM_CATEGORIES][start:end])

    match dist:
        case "mixed":
            save_mixed_data(train_data, val_data, path)
        case "specific":
            save_specific_data(train_data, val_data, path)
        case _:
            raise NotImplementedError(f"{dist} is not implemented")

    return end


def get_three_multi_data(dist: str) -> Dict[str, List[np.ndarray] | np.ndarray]:
    match dist:
        case "mixed":
            DATA_PATH = MULTI_MIXED_DATA_PATH
        case "specific":
            DATA_PATH = MULTI_SPECIFIC_DATA_PATH
        case _:
            raise NotImplementedError(f"{dist} is not implemented")

    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH)

        print("This data downloading process might take a while... be patient.")
        dataset_text = []

        from .utils import WIKI_PATH_DE, WIKI_PATH_IT, WIKI_PATH_FR
        for dataset_idx, dataset_path in zip(["20220301.de", "20220301.it", "20220301.fr"],
                                             [WIKI_PATH_DE, WIKI_PATH_IT, WIKI_PATH_FR]):
            if os.path.isdir(dataset_path):
                print("loading from disk: ", dataset_idx)
                data_one_lang = load_from_disk(dataset_path)
            else:
                data_one_lang = load_dataset("wikipedia", dataset_idx)
                data_one_lang.save_to_disk(dataset_path)
            dataset_text.append(data_one_lang["train"]["text"])

        train_text_per_class = []
        val_text_per_class = []
        print("sample 20% of each dataset")
        for i in range(len(dataset_text)):
            print(f"{i}: {len(dataset_text[i])}")
            sampled_indices = np.random.choice(np.arange(len(dataset_text[i])), size=int(0.2 * len(dataset_text[i])),
                                               replace=False).astype(int)
            dataset_text[i] = [dataset_text[i][ind] for ind in sampled_indices]

            train_size = int(0.84 * len(dataset_text[i]))  # arbitrary split
            train_text_per_class.append(dataset_text[i][:train_size])
            val_text_per_class.append(dataset_text[i][train_size:])

        del dataset_text, sampled_indices

        end = save_train_val_data(dist, train_text_per_class, val_text_per_class, DATA_PATH)
        save_ref_data(DATA_PATH, [val_text_per_class[i][end:] for i in range(NUM_CATEGORIES)])

        print("completed the tokenization process!")

    train_data = []
    val_data = []

    for i in range(MAX_NUM_CLIENTS):
        train_data.append(np.memmap(os.path.join(DATA_PATH, f"train_{i}.bin"), dtype=np.uint16, mode="r"))
        val_data.append(np.memmap(os.path.join(DATA_PATH, f"val_{i}.bin"), dtype=np.uint16, mode="r"))

    ref_data = np.memmap(os.path.join(DATA_PATH, f"ref.bin"), dtype=np.uint16, mode="r")

    return {"train": train_data, "val": val_data, "ref": ref_data}
