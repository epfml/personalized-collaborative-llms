import os

import numpy as np
import tiktoken
import torchtext

AGNEWS_DATA_PATH = os.path.join(os.path.dirname(__file__), "datasets/agnews_mixed/")


# Allocate data to users
def get_agnews_mixed_data():
    if not os.path.exists(AGNEWS_DATA_PATH):
        os.makedirs(AGNEWS_DATA_PATH, exist_ok=True)
        print("downloading data and tokenizing (1-2 min)")
        trainset, testset = torchtext.datasets.AG_NEWS(root=AGNEWS_DATA_PATH + "rawdata")

        trainlabel, traintext = list(zip(*trainset))
        testlabel, testtext = list(zip(*testset))

        tokenizer = tiktoken.get_encoding("gpt2")

        traintext_perclass = []
        testtext_perclass = []
        traintext_perclass_mid = []
        testtext_perclass_mid = []
        trainlabel = np.array(trainlabel)
        testlabel = np.array(testlabel)
        for i in range(1, 5):
            traintext_perclass.append([traintext[ind] for ind in np.where(trainlabel == i)[0]])
            traintext_perclass_mid.append(len(traintext_perclass[i - 1]) // 4)
            testtext_perclass.append([testtext[ind] for ind in np.where(testlabel == i)[0]])
            testtext_perclass_mid.append(len(testtext_perclass[i - 1]) // 4)

        traindata = [
            traintext_perclass[0][:traintext_perclass_mid[0]] + traintext_perclass[1][traintext_perclass_mid[1]:],
            traintext_perclass[0][traintext_perclass_mid[0]:] + traintext_perclass[1][:traintext_perclass_mid[1]],
            traintext_perclass[2][:traintext_perclass_mid[2]] + traintext_perclass[3][traintext_perclass_mid[3]:],
            traintext_perclass[2][traintext_perclass_mid[2]:] + traintext_perclass[3][:traintext_perclass_mid[3]],
        ]
        testdata = [
            testtext_perclass[0][:testtext_perclass_mid[0]] + testtext_perclass[1][testtext_perclass_mid[1]:],
            testtext_perclass[0][testtext_perclass_mid[0]:] + testtext_perclass[1][:testtext_perclass_mid[1]],
            testtext_perclass[2][:testtext_perclass_mid[2]] + testtext_perclass[3][testtext_perclass_mid[3]:],
            testtext_perclass[2][testtext_perclass_mid[2]:] + testtext_perclass[3][:testtext_perclass_mid[3]],
        ]

        for i in range(4):
            traintext = ' '.join(traindata[i])
            testtext = ' '.join(testdata[i])
            raw_tokenized_train = tokenizer.encode_ordinary(traintext)
            raw_tokenized_eval = tokenizer.encode_ordinary(testtext)

            train_tokenized = np.array(raw_tokenized_train, dtype=np.uint16)
            eval_tokenized = np.array(raw_tokenized_eval, dtype=np.uint16)

            train_tokenized.tofile(os.path.join(AGNEWS_DATA_PATH, f'train_{i}.bin'))
            eval_tokenized.tofile(os.path.join(AGNEWS_DATA_PATH, f'val_{i}.bin'))
        print("completed the tokenization process!")

    train_data = []
    val_data = []
    for i in range(4):
        train_data.append(np.memmap(os.path.join(AGNEWS_DATA_PATH, f'train_{i}.bin'), dtype=np.uint16, mode='r'))
        val_data.append(np.memmap(os.path.join(AGNEWS_DATA_PATH, f'val_{i}.bin'), dtype=np.uint16, mode='r'))

    return {'train': train_data, 'val': val_data}
