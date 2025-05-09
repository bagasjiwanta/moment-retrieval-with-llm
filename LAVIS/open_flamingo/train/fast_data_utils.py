import os
import copy
from dataclasses import dataclass
import json
from glob import glob
import random
from typing import Dict, Optional, Sequence, List, Iterator
from operator import itemgetter
from tqdm import tqdm
from typing import Union

import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, DistributedSampler, Sampler
import transformers
from typing import TypedDict, Tuple

from PIL import Image

import open_flamingo.train.conversation as conversation_lib
from open_flamingo.train.data_utils import DataInfo

from open_flamingo.train.any_res_data_utils import process_anyres_image


'''
lazy_dataset = LazySupervisedDataset(...)

output_dir = "/mnt/data/precomputed_samples"
os.makedirs(output_dir, exist_ok=True)

for i in tqdm(range(len(lazy_dataset))):
    sample = lazy_dataset[i]  # already dict of input_ids, labels, images, etc.
    idx = sample.get("qid", i)
    torch.save(sample, os.path.join(output_dir, f"{idx}.pt"))

class PrecomputedDataset(Dataset):
    def __init__(self, directory):
        self.files = sorted(os.listdir(directory))
        self.directory = directory

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        return torch.load(os.path.join(self.directory, self.files[i]))

# Usage
precomp_dataset = PrecomputedDataset("/mnt/data/precomputed_samples")
'''


class FastDataset(Dataset):
    def __init__(self, meta_entries):
        self.entries = meta_entries

    def __getitem__(self, i):
        meta = self.entries[i]
        token_data = torch.load(f"proc_tokens/{meta['id']}.pt")
        images = torch.stack([
            torch.load(f"proc_images/{meta['id']}_img{j}.pt")
            for j in range(meta['num_images'])
        ], dim=0)
        return {
            "input_ids": token_data['input_ids'],
            "labels": token_data['labels'],
            "images": images,
            # ... plus any metadata you want
        }


def pretokenize(raw_dataset):
    # Do ONCE before training
    for entry in raw_dataset:
        token_data = your_tokenizer_function(entry["conversations"])  # returns dict with input_ids, labels
        torch.save(token_data, f"proc_tokens/{entry['id']}.pt")
        for idx, img_path in enumerate(entry["image"]):
            img = image_processor(Image.open(img_path).convert("RGB"))
            torch.save(img, f"proc_images/{entry['id']}_img{idx}.pt")

