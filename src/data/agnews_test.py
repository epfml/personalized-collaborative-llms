import os
from typing import List, Dict

import numpy as np
import tiktoken
import torchtext

TOKENIZER = tiktoken.get_encoding("gpt2")
NUM_CLIENTS = 4
NUM_CATEGORIES = 4

AGNEWS_DATA_PATH = os.path.join(os.path.dirname(__file__), "datasets/agnews")
AGNEWS_TEST_DATA_PATH = os.path.join(AGNEWS_DATA_PATH, "test")

def get_agnews_test_data() -> Dict[str, List[np.ndarray] | np.ndarray]:

    if not os.path.exists(AGNEWS_TEST_DATA_PATH):
        os.makedirs(AGNEWS_TEST_DATA_PATH, exist_ok=True)
        print("downloading data and tokenizing (1-2 min)")
        train_set, val_set = torchtext.datasets.AG_NEWS(root=f"{AGNEWS_DATA_PATH}/raw_data")

        train_label, train_text = list(zip(*train_set))
        val_label, val_text = list(zip(*val_set))

        train_text_per_class = []
        val_text_per_class = []
        train_label = np.array(train_label)
        val_label = np.array(val_label)
        for i in range(1, 5):
            train_text_per_class.append([train_text[ind] for ind in np.where(train_label == i)[0]])
            val_text_per_class.append([val_text[ind] for ind in np.where(val_label == i)[0]])


        for i in range(NUM_CATEGORIES):
            train_text = ' '.join(train_text_per_class[i])
            val_text = ' '.join(val_text_per_class[i])
            raw_tokenized_train = TOKENIZER.encode_ordinary(train_text)
            raw_tokenized_eval = TOKENIZER.encode_ordinary(val_text)

            train_tokenized = np.array(raw_tokenized_train, dtype=np.uint16)
            eval_tokenized = np.array(raw_tokenized_eval, dtype=np.uint16)

            train_tokenized.tofile(os.path.join(AGNEWS_TEST_DATA_PATH, f'train_{i}.bin'))
            eval_tokenized.tofile(os.path.join(AGNEWS_TEST_DATA_PATH, f'val_{i}.bin'))
        print("completed the tokenization process!")

    train_data = []
    val_data = []
    for i in range(NUM_CLIENTS):
        train_data.append(np.memmap(os.path.join(AGNEWS_TEST_DATA_PATH, f'train_{i}.bin'), dtype=np.uint16, mode='r'))
        val_data.append(np.memmap(os.path.join(AGNEWS_TEST_DATA_PATH, f'val_{i}.bin'), dtype=np.uint16, mode='r'))

    return {'train': train_data, 'val': val_data}
