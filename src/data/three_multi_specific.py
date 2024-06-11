import os

import numpy as np
import tiktoken
from datasets import load_from_disk, load_dataset

num_clients = 6
MULTI_DATA_PATH = os.path.join(os.path.dirname(__file__), f"datasets/three_multi_specific/{num_clients}/")


def get_three_multi_data_specific():
    # client's data distribution is fr, de, it, fr, de, it, ....
    # client's number of samples should be different
    samples_size = [250000, 250000, 250000, 50000, 50000, 500000]

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
        for i in range(len(dataset_text)):
            sampled_indices = np.random.choice(np.arange(len(dataset_text[i])), size=int(0.2 * len(dataset_text[i])),
                                               replace=False).astype(int)
            dataset_text[i] = [dataset_text[i][ind] for ind in sampled_indices]
            train_size = int(0.84 * len(dataset_text[i]))
            traintext_perclass.append(dataset_text[i][:train_size])
            testtext_perclass.append(dataset_text[i][train_size:])

        del dataset_text

        traindata = []
        testdata = []
        for i in range(num_clients):
            start = (i // 3) * 1500  # 800000 tokens
            end = ((i // 3) + 1) * 1500
            traindata.append(traintext_perclass[i % 3][start:end])
            start = (i // 3) * 1500  # 160000 tokens
            end = ((i // 3) + 1) * 300
            testdata.append(testtext_perclass[i % 3][start:end])

        for i in range(num_clients):
            traintext = ' '.join(traindata[i])
            testtext = ' '.join(testdata[i])
            raw_tokenized_train = tokenizer.encode_ordinary(traintext)[:samples_size[i]]
            raw_tokenized_eval = tokenizer.encode_ordinary(testtext)[:1000000]

            train_tokenized = np.array(raw_tokenized_train, dtype=np.uint16)
            eval_tokenized = np.array(raw_tokenized_eval, dtype=np.uint16)

            print(f'{i}: {train_tokenized.shape} train, {eval_tokenized.shape} eval ')

            train_tokenized.tofile(os.path.join(MULTI_DATA_PATH, f'train_{i}.bin'))
            eval_tokenized.tofile(os.path.join(MULTI_DATA_PATH, f'val_{i}.bin'))

        del traindata, testdata
        del traintext, testtext, raw_tokenized_eval, raw_tokenized_train, train_tokenized, eval_tokenized
        print("completed the tokenization process!")

    train_data = []
    val_data = []

    for i in range(num_clients):
        train_data.append(np.memmap(os.path.join(MULTI_DATA_PATH, f'train_{i}.bin'), dtype=np.uint16, mode='r'))
        val_data.append(np.memmap(os.path.join(MULTI_DATA_PATH, f'val_{i}.bin'), dtype=np.uint16, mode='r'))


    return {'train': train_data, 'val': val_data, 'samples_size': samples_size}
