import os
from typing import Dict, List

import numpy as np
import tiktoken
from datasets import load_from_disk, load_dataset

MAX_NUM_CLIENTS = 20
DATA_PATH = os.path.join(os.path.dirname(__file__), "datasets/fed_cc_news")
DATASET_PATH = os.path.join(DATA_PATH, "cc_news")


def get_fed_cc_news() -> Dict[str, List[np.ndarray] | np.ndarray]:
    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH)

        print('This data downloading process might take a while... be patient.')
        name = 'cc_news'
        if os.path.isdir(DATASET_PATH):
            print('loading from disk: ', name)
            dataset = load_from_disk(DATASET_PATH)
        else:
            dataset = load_dataset(name, split='train')
            dataset.save_to_disk(DATASET_PATH)

        df = dataset.to_pandas()[['domain', 'text']]
        del dataset

        df = df.groupby('domain').agg({'text': lambda x: x.str.cat(sep=' ')})
        df['len'] = df.text.apply(lambda x: len(x))
        df = df[df.len > 5000000]
        df = df.sample(n=20)

        tokenizer = tiktoken.get_encoding("gpt2")

        ref_data = []
        i = 0
        for index, row in df.iterrows():
            token = tokenizer.encode_ordinary(row['text'])
            total = len(token)
            train_split = int(total * 0.7)
            val_split = int(total * 0.15)
            ref_split = int(total * 0.15)
            raw_tokenized_train = token[:train_split]
            raw_tokenized_eval = token[train_split:train_split + val_split]
            train_tokenized = np.array(raw_tokenized_train, dtype=np.uint16)
            eval_tokenized = np.array(raw_tokenized_eval, dtype=np.uint16)

            raw_tokenized_ref = token[train_split + val_split:train_split + val_split + ref_split]
            ref_data.append(np.array(raw_tokenized_ref, dtype=np.uint16))

            print(f'{index}: {train_tokenized.shape} train, {eval_tokenized.shape} eval ')

            train_tokenized.tofile(os.path.join(DATA_PATH, f'train_{i}.bin'))
            eval_tokenized.tofile(os.path.join(DATA_PATH, f'val_{i}.bin'))
            i = i + 1

        ref_tokenized = np.concatenate(ref_data)
        print(f'{ref_tokenized.shape} ref')
        ref_tokenized.tofile(os.path.join(DATA_PATH, f'ref.bin'))

        del ref_data, raw_tokenized_eval, raw_tokenized_train, raw_tokenized_ref, train_tokenized, eval_tokenized, ref_tokenized
        print("completed the tokenization process!")

    train_data = []
    val_data = []

    for i in range(MAX_NUM_CLIENTS):
        train_data.append(np.memmap(os.path.join(DATA_PATH, f'train_{i}.bin'), dtype=np.uint16, mode='r'))
        val_data.append(np.memmap(os.path.join(DATA_PATH, f'val_{i}.bin'), dtype=np.uint16, mode='r'))

    ref_data = np.memmap(os.path.join(DATA_PATH, f'ref.bin'), dtype=np.uint16, mode='r')

    return {'train': train_data, 'val': val_data, 'ref': ref_data}
