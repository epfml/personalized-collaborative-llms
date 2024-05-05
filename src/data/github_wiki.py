import os
from itertools import islice
from typing import List, Dict

import numpy as np
import tiktoken
from datasets import load_from_disk, load_dataset

TOKENIZER = tiktoken.get_encoding("gpt2")
GIT_DATASET_LOADER = iter(
    load_dataset("codeparrot/github-code-clean", streaming=True, split="train", languages=["Java"]))
MAX_NUM_CLIENTS = 20
NUM_CATEGORIES = 2

GIT_WIKI_DATA_PATH = os.path.join(os.path.dirname(__file__), "datasets/github_wikitext/")
GIT_WIKI_MIXED_DATA_PATH = os.path.join(GIT_WIKI_DATA_PATH, "mixed")
GIT_WIKI_SPECIFIC_DATA_PATH = os.path.join(GIT_WIKI_DATA_PATH, "specific")

MAX_SPECIFIC_TOKENS = {"train": 840000, "val": 160000,
                       "ref": 300000}  # if changed too much, won"t work properly in the splits
MAX_MIXED_TOKENS = {"train": 630000, "val": 160000}
MIN_MIXED_TOKENS = {"train": 210000, "val": 40000}
MIXED_TOKENS_DIST = [
    [3 / 4, 1 / 4],
    [1 / 4, 3 / 4],
]
SPECIFIC_TOKENS_DIST = [
    [1., 0.],
    [0., 1.],
]


def get_github_tokens(incr: int) -> np.ndarray:
    batch = map(lambda x: x["code"], islice(GIT_DATASET_LOADER, incr))
    text = " ".join(batch)
    array = TOKENIZER.encode_ordinary(text)
    tokens = np.array(array, dtype=np.uint16)
    return tokens


def get_github_wikitext_data(dist: str) -> Dict[str, List[np.ndarray] | np.ndarray]:
    match dist:
        case "mixed":
            DATA_PATH = GIT_WIKI_MIXED_DATA_PATH
            l = 1
        case "specific":
            DATA_PATH = GIT_WIKI_SPECIFIC_DATA_PATH
            l = 2
        case _:
            raise NotImplementedError(f"{dist} is not implemented")

    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH)

        print("This data downloading process might take a while... be patient.")

        index = 0

        from .utils import WIKI_PATH_EN
        dataset_idx, dataset_path = "20220301.en", WIKI_PATH_EN
        if os.path.isdir(dataset_path):
            print("loading from disk: ", dataset_idx)
            data_one_lang = load_from_disk(dataset_path)
        else:
            data_one_lang = load_dataset("wikipedia", dataset_idx)
            data_one_lang.save_to_disk(dataset_path)
        dataset_text = data_one_lang["train"]["text"]

        print("sample \% of the data")
        print(f"{len(dataset_text)}")
        sampled_indices = np.random.choice(np.arange(len(dataset_text)), size=int(0.2 * len(dataset_text)),
                                           replace=False).astype(int)
        dataset_text = [dataset_text[ind] for ind in sampled_indices]
        train_size = int(0.84 * len(dataset_text))
        train_text = dataset_text[:train_size]
        val_text = dataset_text[train_size:]

        del dataset_text

        train_data = []
        val_data = []
        end = -1
        for i in range(MAX_NUM_CLIENTS // l):
            start = i * 1500  # 800000 tokens
            end = (i + 1) * 1500
            train_data.append(train_text[start:end])
            start = i * 300  # 160000 tokens
            end = (i + 1) * 300
            val_data.append(val_text[start:end])

        ref_data_wiki = val_text[end: end + 40000]
        ref_text = " ".join(ref_data_wiki)
        raw_tokenized_ref = TOKENIZER.encode_ordinary(ref_text)
        ref_tokenized = np.array(raw_tokenized_ref, dtype=np.uint16)[:500000]
        print(ref_tokenized.shape, "ref wiki")
        git_data = get_github_tokens(incr=50000)
        ref_data_git = git_data[index: index + 500000]
        index += 20000000

        match dist:
            case "mixed":
                for i in range(MAX_NUM_CLIENTS):
                    train_text = " ".join(train_data[i])
                    val_text = " ".join(val_data[i])
                    if i % 2 == 0:
                        raw_tokenized_train = TOKENIZER.encode_ordinary(train_text)[:630000]
                        raw_tokenized_val = TOKENIZER.encode_ordinary(val_text)[:120000]
                        git_train = git_data[index: index + 210000]
                        index += 210000
                        git_val = git_data[index: index + 40000]
                        index += 40000
                    else:
                        raw_tokenized_train = TOKENIZER.encode_ordinary(train_text)[:210000]
                        raw_tokenized_val = TOKENIZER.encode_ordinary(val_text)[:40000]
                        git_train = git_data[index: index + 630000]
                        index += 630000
                        git_val = git_data[index: index + 120000]
                        index += 120000

                    train_tokenized = np.array(raw_tokenized_train, dtype=np.uint16)
                    val_tokenized = np.array(raw_tokenized_val, dtype=np.uint16)
                    train_tokenized = np.concatenate([train_tokenized, git_train])
                    val_tokenized = np.concatenate([val_tokenized, git_val])

                    print(f"{i}: {train_tokenized.shape} train, {val_tokenized.shape} val ")

                    train_tokenized.tofile(os.path.join(DATA_PATH, f"train_{i}.bin"))
                    val_tokenized.tofile(os.path.join(DATA_PATH, f"val_{i}.bin"))
            case "specific":
                for i in range(MAX_NUM_CLIENTS // 2):
                    train_data.append(git_data[index: index + 840000])
                    index += 840000
                    val_data.append(git_data[index: index + 160000])
                    index += 160000

                for i in range(MAX_NUM_CLIENTS // 2):
                    train_text = ' '.join(train_data[i])
                    val_text = ' '.join(val_data[i])
                    raw_tokenized_train = TOKENIZER.encode_ordinary(train_text)[:840000]
                    raw_tokenized_val = TOKENIZER.encode_ordinary(val_text)[:160000]

                    train_tokenized = np.array(raw_tokenized_train, dtype=np.uint16)
                    val_tokenized = np.array(raw_tokenized_val, dtype=np.uint16)

                    print(f'{i}: {train_tokenized.shape} train, {val_tokenized.shape} val ')

                    train_tokenized.tofile(os.path.join(DATA_PATH, f'train_{i}.bin'))
                    val_tokenized.tofile(os.path.join(DATA_PATH, f'val_{i}.bin'))

                for i in range(MAX_NUM_CLIENTS // 2, MAX_NUM_CLIENTS):
                    print(f'{i}: {train_data[i].shape} train, {val_data[i].shape} val ')

                    train_data[i].tofile(os.path.join(DATA_PATH, f'train_{i}.bin'))
                    val_data[i].tofile(os.path.join(DATA_PATH, f'val_{i}.bin'))
            case _:
                raise NotImplementedError(f"{dist} is not implemented")

        ref_tokenized = np.concatenate([ref_tokenized, ref_data_git])
        print(f"{ref_tokenized.shape} ref")

        ref_tokenized.tofile(os.path.join(DATA_PATH, f"ref.bin"))

        del train_data, val_data, ref_data_git, ref_data_wiki
        del train_text, val_text, ref_text, raw_tokenized_val, raw_tokenized_train, raw_tokenized_ref, train_tokenized, val_tokenized, ref_tokenized
        print("completed the tokenization process!")

    train_data = []
    val_data = []
    match dist:
        case "mixed":
            for i in range(MAX_NUM_CLIENTS):
                train_data.append(np.memmap(os.path.join(DATA_PATH, f"train_{i}.bin"), dtype=np.uint16, mode="r"))
                val_data.append(np.memmap(os.path.join(DATA_PATH, f"val_{i}.bin"), dtype=np.uint16, mode="r"))
        case "specific":
            for i in range(0, MAX_NUM_CLIENTS // 2):
                train_data.append(np.memmap(os.path.join(DATA_PATH, f'train_{i}.bin'), dtype=np.uint16, mode='r'))
                val_data.append(np.memmap(os.path.join(DATA_PATH, f'val_{i}.bin'), dtype=np.uint16, mode='r'))
                train_data.append(
                    np.memmap(os.path.join(DATA_PATH, f'train_{i + (MAX_NUM_CLIENTS // 2)}.bin'), dtype=np.uint16,
                              mode='r'))
                val_data.append(
                    np.memmap(os.path.join(DATA_PATH, f'val_{i + (MAX_NUM_CLIENTS // 2)}.bin'), dtype=np.uint16,
                              mode='r'))
        case _:
            raise NotImplementedError(f"{dist} is not implemented")

    ref_data = np.memmap(os.path.join(DATA_PATH, f"ref.bin"), dtype=np.uint16, mode="r")

    return {"train": train_data, "val": val_data, "ref": ref_data}
