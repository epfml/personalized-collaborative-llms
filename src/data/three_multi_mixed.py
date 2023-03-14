import os

import numpy as np
import tiktoken
from datasets import load_from_disk, load_dataset

num_clients = 20
MULTI_DATA_PATH = os.path.join(os.path.dirname(__file__), f"datasets/three_multi_mixed/{num_clients}/")

def get_three_multi_data_mixed():

    if not os.path.exists(MULTI_DATA_PATH):
        os.makedirs(MULTI_DATA_PATH)

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
        print('sample \% of the data')
        for i in range(len(dataset_text)):
            print(f'{i}: {len(dataset_text[i])}')
            sampled_indices = np.random.choice(np.arange(len(dataset_text[i])), size=int(0.2 * len(dataset_text[i])),
                                           replace=False).astype(int)
            dataset_text[i] = [dataset_text[i][ind] for ind in sampled_indices]
            train_size = int(0.84 * len(dataset_text[i]))
            traintext_perclass.append(dataset_text[i][:train_size])
            print(f'Train length {i}: {sum(map(lambda x: len(x), traintext_perclass[i]))}')
            testtext_perclass.append(dataset_text[i][train_size:])
            print(f'Test length {i}: {sum(map(lambda x: len(x), testtext_perclass[i]))}')

        del dataset_text

        traindata = []
        testdata = []
        ref_data = []
        for i in range(num_clients * 2):
            start = (i // 3) * 1500 # 800000 tokens
            end = ((i // 3) + 1) * 1500
            traindata.append(traintext_perclass[i % 3][start:end])
            start = (i // 3) * 300 # 160000 tokens
            end = ((i // 3) + 1) * 300
            testdata.append(testtext_perclass[i % 3][start:end])

        for i in range(num_clients, num_clients + 3):
            diff = (i // 3) * 2000  # 1600000 tokens
            reftext = ' '.join(testtext_perclass[i % 3][end:end + diff])
            raw_tokenized_ref = tokenizer.encode_ordinary(reftext)
            ref_data.append(np.array(raw_tokenized_ref, dtype=np.uint16)[:300000])

        for i in range(num_clients):
            traintext = ' '.join(traindata[i])
            testtext = ' '.join(testdata[i])
            raw_tokenized_train = tokenizer.encode_ordinary(traintext)[:630000]
            raw_tokenized_eval = tokenizer.encode_ordinary(testtext)[:120000]

            traintext = ' '.join(traindata[num_clients + ((i + 1) % num_clients)])
            testtext = ' '.join(testdata[num_clients + ((i + 1) % num_clients)])
            raw_tokenized_train.extend(tokenizer.encode_ordinary(traintext)[:210000])
            raw_tokenized_eval.extend(tokenizer.encode_ordinary(testtext)[:40000])

            train_tokenized = np.array(raw_tokenized_train, dtype=np.uint16)
            eval_tokenized = np.array(raw_tokenized_eval, dtype=np.uint16)

            print(f'{i}: {train_tokenized.shape} train, {eval_tokenized.shape} eval ')

            train_tokenized.tofile(os.path.join(MULTI_DATA_PATH, f'train_{i}.bin'))
            eval_tokenized.tofile(os.path.join(MULTI_DATA_PATH, f'val_{i}.bin'))

        ref_tokenized = np.concatenate(ref_data)

        print(f'{ref_tokenized.shape} ref')

        ref_tokenized.tofile(os.path.join(MULTI_DATA_PATH, f'ref.bin'))

        del traindata, testdata, ref_data
        del traintext, testtext, reftext, raw_tokenized_eval, raw_tokenized_train, raw_tokenized_ref, train_tokenized, eval_tokenized, ref_tokenized
        print("completed the tokenization process!")

    train_data = []
    val_data = []

    for i in range(num_clients):
        train_data.append(np.memmap(os.path.join(MULTI_DATA_PATH, f'train_{i}.bin'), dtype=np.uint16, mode='r'))
        val_data.append(np.memmap(os.path.join(MULTI_DATA_PATH, f'val_{i}.bin'), dtype=np.uint16, mode='r'))

    ref_data = np.memmap(os.path.join(MULTI_DATA_PATH, f'ref.bin'), dtype=np.uint16, mode='r')

    return {'train': train_data, 'val': val_data, 'ref': ref_data}