import os
from itertools import islice

import numpy as np
import tiktoken
from datasets import load_from_disk, load_dataset


WIKI_PATH_EN = os.path.join(os.path.dirname(__file__), "wikipedia/en")

num_clients = 20
MULTI_DATA_PATH = os.path.join(os.path.dirname(__file__), f"datasets/github_wikitext_mixed/{num_clients}")
ds = load_dataset("codeparrot/github-code-clean", streaming=True, split="train", languages=["Java"])
ds = iter(ds)
tokenizer = tiktoken.get_encoding("gpt2")


def get_github_tokens(incr):
    batch = map(lambda x: x['code'], islice(ds, incr))
    text = ' '.join(batch)
    arry = tokenizer.encode_ordinary(text)
    tokens = np.array(arry, dtype=np.uint16)
    return tokens


def get_github_wikitext_data_mixed():
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
        for i in range(num_clients):
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

        for i in range(num_clients):
            traintext = ' '.join(traindata[i])
            testtext = ' '.join(testdata[i])
            if num_clients % 2 == 0:
                raw_tokenized_train = tokenizer.encode_ordinary(traintext)[:630000]
                raw_tokenized_eval = tokenizer.encode_ordinary(testtext)[:120000]
                git_train = git_data[index: index + 210000]
                index += 210000
                git_test = git_data[index: index + 40000]
                index += 40000
            else:
                raw_tokenized_train = tokenizer.encode_ordinary(traintext)[:210000]
                raw_tokenized_eval = tokenizer.encode_ordinary(testtext)[:40000]
                git_train = git_data[index: index + 630000]
                index += 630000
                git_test = git_data[index: index + 120000]
                index += 120000

            train_tokenized = np.array(raw_tokenized_train, dtype=np.uint16)
            eval_tokenized = np.array(raw_tokenized_eval, dtype=np.uint16)
            train_tokenized = np.concatenate([train_tokenized, git_train])
            eval_tokenized = np.concatenate([eval_tokenized, git_test])

            print(f'{i}: {train_tokenized.shape} train, {eval_tokenized.shape} eval ')

            train_tokenized.tofile(os.path.join(MULTI_DATA_PATH, f'train_{i}.bin'))
            eval_tokenized.tofile(os.path.join(MULTI_DATA_PATH, f'val_{i}.bin'))

        ref_tokenized = np.concatenate([ref_tokenized, refdata_git])
        print(f'{ref_tokenized.shape} ref')

        ref_tokenized.tofile(os.path.join(MULTI_DATA_PATH, f'ref.bin'))

        del traindata, testdata, refdata_git, refdata_wiki
        del traintext, testtext, reftext, raw_tokenized_eval, raw_tokenized_train, raw_tokenized_ref, train_tokenized, eval_tokenized, ref_tokenized
        print("completed the tokenization process!")

    train_data = []
    val_data = []
    for i in range(num_clients):
        train_data.append(np.memmap(os.path.join(MULTI_DATA_PATH, f'train_{i}.bin'), dtype=np.uint16, mode='r'))
        val_data.append(np.memmap(os.path.join(MULTI_DATA_PATH, f'val_{i}.bin'), dtype=np.uint16, mode='r'))

    ref_data = np.memmap(os.path.join(MULTI_DATA_PATH, f'ref.bin'), dtype=np.uint16, mode='r')

    return {'train': train_data, 'val': val_data, 'ref': ref_data}
