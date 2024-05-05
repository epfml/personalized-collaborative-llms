import os
from typing import Dict, List

import numpy as np
import tiktoken
from datasets import load_from_disk, load_dataset

from .utils import WIKI_PATH_EN, WIKI_PATH_DE, WIKI_PATH_FR, WIKI_PATH_IT

SPLIT_DATA_PATH = os.path.join(os.path.dirname(__file__), f"datasets/split_wiki/")
MAX_TRAIN_TOKEN = 84000000
MAX_TEST_TOKEN = 16000000


def get_split_multi_data(language: str) -> Dict[str, List[np.ndarray]]:
    path = os.path.join(SPLIT_DATA_PATH, language)

    if not os.path.exists(path):
        os.makedirs(path)

        print("This data downloading process might take a while... be patient.")

        dataset_idx = f"20220301.{language}"
        match language:
            case "en":
                dataset_path = WIKI_PATH_EN
            case "de":
                dataset_path = WIKI_PATH_DE
            case "fr":
                dataset_path = WIKI_PATH_FR
            case "it":
                dataset_path = WIKI_PATH_IT
            case _:
                raise NotImplementedError

        if os.path.isdir(dataset_path):
            print("loading from disk: ", dataset_idx)
            data_one_lang = load_from_disk(dataset_path)
        else:
            data_one_lang = load_dataset("wikipedia", dataset_idx)
            data_one_lang.save_to_disk(dataset_path)
        dataset_text = data_one_lang["train"]["text"]

        tokenizer = tiktoken.get_encoding("gpt2")

        print("sample 10% of the data")  # 10 % is a sufficient chunk
        sampled_indices = np.random.choice(np.arange(len(dataset_text)), size=int(0.1 * len(dataset_text)),
                                           replace=False).astype(int)
        dataset_text = [dataset_text[ind] for ind in sampled_indices]

        train_size = int(0.84 * len(dataset_text))  # arbitrary split at 84 %
        train_text = dataset_text[:train_size]
        test_text = dataset_text[train_size:]

        del data_one_lang, dataset_text

        train_text = " ".join(train_text)
        test_text = " ".join(test_text)
        raw_tokenized_train = tokenizer.encode_ordinary(train_text)[:MAX_TRAIN_TOKEN]
        raw_tokenized_eval = tokenizer.encode_ordinary(test_text)[:MAX_TEST_TOKEN]

        train_tokenized = np.array(raw_tokenized_train, dtype=np.uint16)
        eval_tokenized = np.array(raw_tokenized_eval, dtype=np.uint16)

        print(f"{train_tokenized.shape} train, {eval_tokenized.shape} eval ")

        train_tokenized.tofile(os.path.join(path, f"train.bin"))
        eval_tokenized.tofile(os.path.join(path, f"val.bin"))

        del train_text, test_text, raw_tokenized_eval, raw_tokenized_train, train_tokenized, eval_tokenized
        print("completed the tokenization process!")

    train_data = np.memmap(os.path.join(path, f"train.bin"), dtype=np.uint16, mode="r")
    val_data = np.memmap(os.path.join(path, f"val.bin"), dtype=np.uint16, mode="r")

    return {"train": [train_data], "val": [val_data]}
