import os

import numpy as np
import tiktoken
from datasets import load_from_disk, load_dataset

num_clients = 9
MULTI_DATA_PATH = os.path.join(os.path.dirname(__file__), f"datasets/three_multi_specific/{num_clients}/")

def get_min_token(idx, text_data, min_token, tokenizer):
    data = []
    incr = min_token * 3
    while len(data) < min_token:
        print(idx, len(data))
        data.extend(tokenizer.encode_ordinary(' '.join(text_data)[idx:idx + incr]))
        idx += incr

    return idx, data

def get_three_multi_data_specific(token_train_size: int):
    NEW_MULTI_DATA_PATH = os.path.join(MULTI_DATA_PATH, f"{token_train_size}/")

    if not os.path.exists(NEW_MULTI_DATA_PATH):
        os.makedirs(NEW_MULTI_DATA_PATH)

        print('This data downloading process might take a while... be patient.')
        dataset_text = []

        for dataset_idx in ["20220301.de", "20220301.it",
                            "20220301.fr"]:
            if os.path.isdir(dataset_idx):
                print('loading from disk: ', dataset_idx)
                data_one_lang = load_from_disk(dataset_idx)
            else:
                data_one_lang = load_dataset("wikipedia", dataset_idx)
                data_one_lang.save_to_disk(dataset_idx)
            dataset_text.append(data_one_lang['train']['text'])

        tokenizer = tiktoken.get_encoding("gpt2")

        traintext_perclass = []
        testtext_perclass = []
        print(f'sample \% of the data, {token_train_size}')
        for i in range(len(dataset_text)):
            print(f'{i}: {len(dataset_text[i])}')
            sampled_indices = np.random.choice(np.arange(len(dataset_text[i])), size=int(0.5 * len(dataset_text[i])),
                                           replace=False).astype(int)
            dataset_text[i] = [dataset_text[i][ind] for ind in sampled_indices]
            train_size = int(0.95 * len(dataset_text[i]))
            traintext_perclass.append(dataset_text[i][:train_size])
            print(f'Train length {i}: {sum(map(lambda x: len(x), traintext_perclass[i]))}')
            testtext_perclass.append(dataset_text[i][train_size:])
            print(f'Test length {i}: {sum(map(lambda x: len(x), testtext_perclass[i]))}')

        del dataset_text

        ref_data = []
        train_idx = [0] * 3
        test_idx = [0] * 3
        for i in range(num_clients):
            train_idx[i % 3], data = get_min_token(train_idx[i % 3], traintext_perclass[i % 3], token_train_size, tokenizer)
            train_tokenized = np.array(data[:token_train_size], dtype=np.uint16)
            train_tokenized.tofile(os.path.join(NEW_MULTI_DATA_PATH, f'train_{i}.bin'))

            test_idx[i % 3], data = get_min_token(test_idx[i % 3], testtext_perclass[i % 3], 160000, tokenizer)
            eval_tokenized = np.array(data[:160000], dtype=np.uint16)
            eval_tokenized.tofile(os.path.join(NEW_MULTI_DATA_PATH, f'val_{i}.bin'))

            print(f'{i}: {train_tokenized.shape} train, {eval_tokenized.shape} eval ')

        for i in range(num_clients, num_clients + 3):
            _, data = get_min_token(test_idx[i % 3], testtext_perclass[i % 3], 300000, tokenizer)
            ref_data.append(np.array(data, dtype=np.uint16)[:300000])

        ref_tokenized = np.concatenate(ref_data)

        print(f'{ref_tokenized.shape} ref')

        ref_tokenized.tofile(os.path.join(NEW_MULTI_DATA_PATH, f'ref.bin'))

        del ref_data
        del train_tokenized, eval_tokenized, ref_tokenized
        print("completed the tokenization process!")

    train_data = []
    val_data = []

    for i in range(num_clients):
        train_data.append(np.memmap(os.path.join(NEW_MULTI_DATA_PATH, f'train_{i}.bin'), dtype=np.uint16, mode='r'))
        val_data.append(np.memmap(os.path.join(NEW_MULTI_DATA_PATH, f'val_{i}.bin'), dtype=np.uint16, mode='r'))

    ref_data = np.memmap(os.path.join(NEW_MULTI_DATA_PATH, f'ref.bin'), dtype=np.uint16, mode='r')

    return {'train': train_data, 'val': val_data, 'ref': ref_data}