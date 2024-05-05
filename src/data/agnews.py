import os
from typing import List, Dict

import numpy as np
import tiktoken
import torchtext

TOKENIZER = tiktoken.get_encoding("gpt2")
NUM_CLIENTS_MIXED = 4
NUM_CLIENTS_SPECIFIC = 8
NUM_CATEGORIES = 4

AGNEWS_DATA_PATH = os.path.join(os.path.dirname(__file__), "datasets/agnews")
AGNEWS_MIXED_DATA_PATH = os.path.join(AGNEWS_DATA_PATH, "mixed")
AGNEWS_SPECIFIC_DATA_PATH = os.path.join(AGNEWS_DATA_PATH, "specific")

MIXED_TOKENS_DIST = [
    [3 / 4, 1 / 4, 0., 0.],
    [1 / 4, 3 / 4, 0., 0.],
    [0., 0., 3 / 4, 1 / 4],
    [0., 0., 1 / 4, 3 / 4],
]
SPECIFIC_TOKENS_DIST = [
    [1., 0., 0., 0.],
    [0., 1., 0., 0.],
    [0., 0., 1., 0.],
    [0., 0., 0., 1.],
]


def get_agnews_data(dist: str) -> Dict[str, List[np.ndarray] | np.ndarray]:
    match dist:
        case "mixed":
            DATA_PATH = AGNEWS_MIXED_DATA_PATH
            c = NUM_CLIENTS_MIXED
        case "specific":
            DATA_PATH = AGNEWS_SPECIFIC_DATA_PATH
            c = NUM_CLIENTS_SPECIFIC
        case _:
            raise NotImplementedError(f"{dist} is not implemented")

    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH, exist_ok=True)
        print("downloading data and tokenizing (1-2 min)")
        train_set, val_set = torchtext.datasets.AG_NEWS(root=f"{AGNEWS_DATA_PATH}/raw_data")

        train_label, train_text = list(zip(*train_set))
        val_label, val_text = list(zip(*val_set))

        tokenizer = tiktoken.get_encoding("gpt2")

        train_text_per_class = []
        val_text_per_class = []
        train_text_per_class_mid = []
        val_text_per_class_mid = []
        train_label = np.array(train_label)
        val_label = np.array(val_label)

        match dist:
            case "mixed":
                l = 4
            case "specific":
                l = 2
            case _:
                raise NotImplementedError(f"Not implemented {dist}")

        for i in range(1, 5):
            train_text_per_class.append([train_text[ind] for ind in np.where(train_label == i)[0]])
            train_text_per_class_mid.append(len(train_text_per_class[i - 1]) // l)
            val_text_per_class.append([val_text[ind] for ind in np.where(val_label == i)[0]])
            val_text_per_class_mid.append(len(val_text_per_class[i - 1]) // l)

        match dist:
            case "mixed":
                train_data = [
                    train_text_per_class[0][:train_text_per_class_mid[0]] + train_text_per_class[1][
                                                                            train_text_per_class_mid[1]:],
                    train_text_per_class[0][train_text_per_class_mid[0]:] + train_text_per_class[1][
                                                                            :train_text_per_class_mid[1]],
                    train_text_per_class[2][:train_text_per_class_mid[2]] + train_text_per_class[3][
                                                                            train_text_per_class_mid[3]:],
                    train_text_per_class[2][train_text_per_class_mid[2]:] + train_text_per_class[3][
                                                                            :train_text_per_class_mid[3]],
                ]
                val_data = [
                    val_text_per_class[0][:val_text_per_class_mid[0]] + val_text_per_class[1][
                                                                        val_text_per_class_mid[1]:],
                    val_text_per_class[0][val_text_per_class_mid[0]:] + val_text_per_class[1][
                                                                        :val_text_per_class_mid[1]],
                    val_text_per_class[2][:val_text_per_class_mid[2]] + val_text_per_class[3][
                                                                        val_text_per_class_mid[3]:],
                    val_text_per_class[2][val_text_per_class_mid[2]:] + val_text_per_class[3][
                                                                        :val_text_per_class_mid[3]],
                ]
            case "specific":
                train_data = [
                    train_text_per_class[0][:train_text_per_class_mid[0]],
                    train_text_per_class[0][train_text_per_class_mid[0]:],
                    train_text_per_class[1][:train_text_per_class_mid[0]],
                    train_text_per_class[1][train_text_per_class_mid[0]:],
                    train_text_per_class[2][:train_text_per_class_mid[0]],
                    train_text_per_class[2][train_text_per_class_mid[0]:],
                    train_text_per_class[3][:train_text_per_class_mid[0]],
                    train_text_per_class[3][train_text_per_class_mid[0]:],
                ]
                val_data = [
                    val_text_per_class[0][:val_text_per_class_mid[0]],
                    val_text_per_class[0][val_text_per_class_mid[0]:],
                    val_text_per_class[1][:val_text_per_class_mid[0]],
                    val_text_per_class[1][val_text_per_class_mid[0]:],
                    val_text_per_class[2][:val_text_per_class_mid[0]],
                    val_text_per_class[2][val_text_per_class_mid[0]:],
                    val_text_per_class[3][:val_text_per_class_mid[0]],
                    val_text_per_class[3][val_text_per_class_mid[0]:],
                ]
            case _:
                raise NotImplementedError(f"Not implemented {dist}")

        for i in range(c):
            train_text = ' '.join(train_data[i])
            val_text = ' '.join(val_data[i])
            raw_tokenized_train = tokenizer.encode_ordinary(train_text)
            raw_tokenized_eval = tokenizer.encode_ordinary(val_text)

            train_tokenized = np.array(raw_tokenized_train, dtype=np.uint16)
            eval_tokenized = np.array(raw_tokenized_eval, dtype=np.uint16)

            train_tokenized.tofile(os.path.join(DATA_PATH, f'train_{i}.bin'))
            eval_tokenized.tofile(os.path.join(DATA_PATH, f'val_{i}.bin'))
        print("completed the tokenization process!")

    train_data = []
    val_data = []
    for i in range(c):
        train_data.append(np.memmap(os.path.join(DATA_PATH, f'train_{i}.bin'), dtype=np.uint16, mode='r'))
        val_data.append(np.memmap(os.path.join(DATA_PATH, f'val_{i}.bin'), dtype=np.uint16, mode='r'))

    return {'train': train_data, 'val': val_data}
