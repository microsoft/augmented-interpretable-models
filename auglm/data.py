import torchhd
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.func import jacfwd
from torch.autograd.functional import jacobian
from tqdm import tqdm
import torch
from torch.autograd import grad
from dict_hash import sha256
from os.path import join
import transformers
import joblib
import pandas as pd
import scipy
import os
import numpy as np
from copy import deepcopy
import sys
import datasets
import matplotlib.pyplot as plt
import numpy as np
sys.path.append('../experiments/')

BABYLM_ROOT_DIR = os.path.expanduser(
    f'~/augmented-interpretable-models/babylm_data/')


class BabyLMDataset(Dataset):

    def __init__(self, tokens_file, max_length):
        self.tokens = joblib.load(tokens_file)
        self.max_length = max_length

    def __len__(self):
        return len(self.tokens) - self.max_length

    def __getitem__(self, idx):
        return torch.tensor(self.tokens[idx: idx + self.max_length]), torch.tensor(self.tokens[idx + self.max_length])


if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    # # preprocess
    # for k in ['dev', 'test']:
    #     BABY_LM_DIR = join(BABYLM_ROOT_DIR, f'babylm_{k}')
    #     files = [join(BABY_LM_DIR, f)
    #              for f in os.listdir(BABY_LM_DIR) if f.endswith(f'.{k}')]
    #     texts = '\n\n'.join([open(f).read() for f in files])
    #     tokens = []
    #     chunk_size = 10000
    #     for i in tqdm(range(0, len(texts), chunk_size)):
    #         tokens += tokenizer(texts[i:i+chunk_size])['input_ids']
    #     joblib.dump(tokens, join(BABY_LM_DIR, 'full.joblib'))
