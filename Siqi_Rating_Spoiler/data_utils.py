# Adapted from https://github.com/yzhan238/PIEClass/blob/main/src/data_utils.py

from joblib import Parallel, delayed
from sklearn.metrics import f1_score
import torch
import os
from math import ceil
from tqdm import tqdm
import pickle
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from collections import defaultdict


def acc(pred, gt):
    return f1_score(gt, pred, average='micro'), f1_score(gt, pred, average='macro')

def encode(docs, tokenizer, max_len):
    encoded_dict = tokenizer.batch_encode_plus(docs, add_special_tokens=True, 
                                               max_length=max_len, padding='max_length',
                                                return_attention_mask=True, truncation=True, 
                                               return_tensors='pt')
    input_ids = encoded_dict['input_ids']
    attention_masks = encoded_dict['attention_mask']
    return input_ids, attention_masks

def create_dataset(docs, labels, tokenizer, loader_name, max_len=512, num_cpus=20):
    if os.path.exists(loader_name):
        print(f"Loading encoded texts from {loader_name}")
        data = torch.load(loader_name)
    else:
        print(f"Converting texts into tensors.")
        chunk_size = ceil(len(docs) / num_cpus)
        chunks = [docs[x:x+chunk_size] for x in range(0, len(docs), chunk_size)]
        results = Parallel(n_jobs=num_cpus)(delayed(encode)(docs=chunk, tokenizer=tokenizer, max_len=max_len) for chunk in chunks)
        input_ids = torch.cat([result[0] for result in results])
        attention_masks = torch.cat([result[1] for result in results])
        labels = torch.tensor(labels)
        data = {"input_ids": input_ids, "attention_masks": attention_masks, "labels": labels}
        print(f"Saving encoded texts into {loader_name}")
        torch.save(data, loader_name)
    return data

def make_dataloader(data_dict, batch_size, ids=None, shuffle=False):
    if ids:
        dataset = TensorDataset(data_dict["input_ids"][ids], data_dict["attention_masks"][ids], data_dict["labels"][ids])
    else:
        dataset = TensorDataset(data_dict["input_ids"], data_dict["attention_masks"], data_dict["labels"])
    dataset_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataset_loader

def up_sample(labels):
    split = defaultdict(list)
    for i, l in enumerate(labels):
        split[l].append(i)
    max_len = max([len(ids) for ids in split.values()])
    new_ids = []
    for ids in split.values():
        copy = max_len // len(ids)
        for _ in range(copy):
            new_ids.extend(ids)
        new_ids.extend(np.random.choice(ids, size=max_len%len(ids), replace=False).tolist())
    return new_ids

def down_sample(labels):
    split = defaultdict(list)
    for i, l in enumerate(labels):
        split[l].append(i)
    min_len = min([len(ids) for ids in split.values()])
    new_ids = []
    for ids in split.values():
        new_ids.extend(np.random.choice(ids, size=min_len, replace=False).tolist())
    return new_ids