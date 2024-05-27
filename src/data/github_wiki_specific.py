import os
from itertools import islice

import numpy as np
import tiktoken
from datasets import load_from_disk, load_dataset

num_clients = 20
MULTI_DATA_PATH = os.path.join(os.path.dirname(__file__), f"datasets/github_wikitext_specific/{num_clients}")
ds = load_dataset("codeparrot/github-code-clean", streaming=True, split="train", languages=["Java"])
ds = iter(ds)
tokenizer = tiktoken.get_encoding("gpt2")


def get_github_tokens(incr):
    batch = map(lambda x: x['code'], islice(ds, incr))
    text = ' '.join(batch)
    arry = tokenizer.encode_ordinary(text)
    tokens = np.array(arry, dtype=np.uint16)
    return tokens


def get_github_wikitext_data_specific():
    if not os.path.exists(MULTI_DATA_PATH):
        os.makedirs(MULTI_DATA_PATH)

        print('This data downloading process might take a while... be patient.')

        index = 0

        dataset_idx = "20220301.en"
        if os.path.isdir(dataset_idx):
            print('loading from disk: ', dataset_idx)
            data_one_lang = load_from_disk(dataset_idx)
        else:
            data_one_lang = load_dataset("wikipedia", dataset_idx)
            data_one_lang.save_to_disk(dataset_idx)
        dataset_text = data_one_lang['train']['text']

        print('sample \% of the data')
        print(f'{len(dataset_text)}')
        sampled_indices = np.random.choice(np.arange(len(dataset_text)), size=int(0.2 * len(dataset_text)),
                                           replace=False).astype(int)
        dataset_text = [dataset_text[ind] for ind in sampled_indices]
        train_size = int(0.84 * len(dataset_text))
        traintext = dataset_text[:train_size]
        print(f'Train length: {sum(map(lambda x: len(x), traintext))}')
        testtext = dataset_text[train_size:]
        print(f'Test length: {sum(map(lambda x: len(x), testtext))}')

        del dataset_text

        traindata = []
        testdata = []
        end = -1
        for i in range(num_clients // 2):
            start = i * 1500  # 800000 tokens
            end = (i + 1) * 1500
            traindata.append(traintext[start:end])
            start = i * 300  # 160000 tokens
            end = (i + 1) * 300
            testdata.append(testtext[start:end])

        refdata_wiki = testtext[end: end + 40000]
        reftext = ' '.join(refdata_wiki)
        raw_tokenized_ref = tokenizer.encode_ordinary(reftext)
        ref_tokenized = np.array(raw_tokenized_ref, dtype=np.uint16)[:500000]
        print(ref_tokenized.shape, 'ref wiki')
        git_data = get_github_tokens(incr=50000)
        refdata_git = git_data[index: index + 500000]
        index += 20000000

        for i in range(num_clients // 2):
            traindata.append(git_data[index: index + 840000])
            index += 840000
            testdata.append(git_data[index: index + 160000])
            index += 160000

        for i in range(num_clients // 2):
            traintext = ' '.join(traindata[i])
            testtext = ' '.join(testdata[i])
            raw_tokenized_train = tokenizer.encode_ordinary(traintext)[:840000]
            raw_tokenized_eval = tokenizer.encode_ordinary(testtext)[:160000]

            train_tokenized = np.array(raw_tokenized_train, dtype=np.uint16)
            eval_tokenized = np.array(raw_tokenized_eval, dtype=np.uint16)

            print(f'{i}: {train_tokenized.shape} train, {eval_tokenized.shape} eval ')

            train_tokenized.tofile(os.path.join(MULTI_DATA_PATH, f'train_{i}.bin'))
            eval_tokenized.tofile(os.path.join(MULTI_DATA_PATH, f'val_{i}.bin'))

        for i in range(num_clients // 2, num_clients):
            print(f'{i}: {traindata[i].shape} train, {testdata[i].shape} eval ')

            traindata[i].tofile(os.path.join(MULTI_DATA_PATH, f'train_{i}.bin'))
            testdata[i].tofile(os.path.join(MULTI_DATA_PATH, f'val_{i}.bin'))

        ref_tokenized = np.concatenate([ref_tokenized, refdata_git])
        print(f'{ref_tokenized.shape} ref')

        ref_tokenized.tofile(os.path.join(MULTI_DATA_PATH, f'ref.bin'))

        del traindata, testdata, refdata_git, refdata_wiki
        del traintext, testtext, reftext, raw_tokenized_eval, raw_tokenized_train, raw_tokenized_ref, train_tokenized, eval_tokenized, ref_tokenized
        print("completed the tokenization process!")

    train_data = []
    val_data = []
    for i in range(0, num_clients // 2):
        train_data.append(np.memmap(os.path.join(MULTI_DATA_PATH, f'train_{i}.bin'), dtype=np.uint16, mode='r'))
        val_data.append(np.memmap(os.path.join(MULTI_DATA_PATH, f'val_{i}.bin'), dtype=np.uint16, mode='r'))
        train_data.append(
            np.memmap(os.path.join(MULTI_DATA_PATH, f'train_{i + (num_clients // 2)}.bin'), dtype=np.uint16, mode='r'))
        val_data.append(
            np.memmap(os.path.join(MULTI_DATA_PATH, f'val_{i + (num_clients // 2)}.bin'), dtype=np.uint16, mode='r'))

    ref_data = np.memmap(os.path.join(MULTI_DATA_PATH, f'ref.bin'), dtype=np.uint16, mode='r')

    return {'train': train_data, 'val': val_data, 'ref': ref_data}
