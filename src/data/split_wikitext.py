import os

import numpy as np
import tiktoken
from datasets import load_from_disk, load_dataset

MULTI_DATA_PATH = os.path.join(os.path.dirname(__file__), f"datasets/split_wiki/")


def get_split_multi_data(language: str):
    path = os.path.join(MULTI_DATA_PATH, language)

    if not os.path.exists(path):
        os.makedirs(path)

        print('This data downloading process might take a while... be patient.')

        dataset_idx = f'20220301.{language}'
        if os.path.isdir(dataset_idx):
            print('loading from disk: ', dataset_idx)
            data_one_lang = load_from_disk(dataset_idx)
        else:
            data_one_lang = load_dataset("wikipedia", dataset_idx)
            data_one_lang.save_to_disk(dataset_idx)
        dataset_text = data_one_lang['train']['text']

        tokenizer = tiktoken.get_encoding("gpt2")

        print('sample \% of the data')
        sampled_indices = np.random.choice(np.arange(len(dataset_text)), size=int(0.1 * len(dataset_text)),
                                           replace=False).astype(int)
        dataset_text = [dataset_text[ind] for ind in sampled_indices]
        train_size = int(0.84 * len(dataset_text))
        train_text = dataset_text[:train_size]
        print(f'Train length: {sum(map(lambda x: len(x), train_text))}')
        test_text = dataset_text[train_size:]
        print(f'Test length: {sum(map(lambda x: len(x), test_text))}')

        del data_one_lang, dataset_text

        train_text = ' '.join(train_text)
        test_text = ' '.join(test_text)
        raw_tokenized_train = tokenizer.encode_ordinary(train_text)[:84000000]
        raw_tokenized_eval = tokenizer.encode_ordinary(test_text)[:16000000]

        train_tokenized = np.array(raw_tokenized_train, dtype=np.uint16)
        eval_tokenized = np.array(raw_tokenized_eval, dtype=np.uint16)

        print(f'{train_tokenized.shape} train, {eval_tokenized.shape} eval ')

        train_tokenized.tofile(os.path.join(path, f'train_{language}.bin'))
        eval_tokenized.tofile(os.path.join(path, f'val_{language}.bin'))

        del train_text, test_text, raw_tokenized_eval, raw_tokenized_train, train_tokenized, eval_tokenized
        print("completed the tokenization process!")

    train_data = np.memmap(os.path.join(path, f'train_{language}.bin'), dtype=np.uint16, mode='r')
    val_data = np.memmap(os.path.join(path, f'val_{language}.bin'), dtype=np.uint16, mode='r')

    return {'train': train_data, 'val': val_data}
