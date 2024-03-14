from typing import List
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

# get path to current file
path_to_file = os.path.dirname(os.path.abspath(__file__))
BABYLM_ROOT_DIR = join(path_to_file, 'babylm_data')
TEXT_MINI = '''In the quiet village of Willow Creek, nestled between rolling hills and whispering woods, lived an old gardener named Eli. With hands weathered by time and soil, Eli tended to his garden with a love so deep it made the flowers blush and the trees stand a bit taller. One spring morning, as the first rays of sunshine pierced the dewy air, Eli discovered a rare blue rose blooming among the sea of greens and colors—a rose he had heard of in stories but never believed to exist. This miraculous find became the talk of the village, drawing curious eyes and eager hearts to Eli's garden. But in his wisdom, Eli knew the rose wasn't meant for fame or fortune; it was a reminder from the earth, a symbol of the beauty and mystery that lies in the simplest moments of life. He cared for the blue rose, letting it thrive in its natural home, while continuing his daily rituals, teaching all who visited that the truest treasures are often hidden in plain sight, nurtured by patience and love.'''


class NextWordDataset(Dataset):
    '''This class is used to create a dataset for the next word prediction task.
    It returns tensors of token indexes (numbers) that can be decoded and inspected with the tokenizer.
    '''

    def __init__(self, tokens_file: str = None, raw_tokens: List = None, max_n_tokens=32):
        self.max_n_tokens = max_n_tokens
        if tokens_file is None and raw_tokens is None:
            raise ValueError(
                'Either tokens_file or raw_tokens must be provided')
        if raw_tokens is not None:
            self.tokens_ = raw_tokens
        else:
            self.tokens_ = joblib.load(tokens_file)

    def __len__(self):
        return len(self.tokens_) - self.max_n_tokens

    def __getitem__(self, idx):
        return torch.tensor(self.tokens_[idx: idx + self.max_n_tokens]), torch.tensor(self.tokens_[idx + self.max_n_tokens])


def preprocess_babylm_data(tokenizer_checkpoint='gpt2', BABYLM_ROOT_DIR=BABYLM_ROOT_DIR):
    '''Tokenizes all the babylm text and dumps it into a joblib file
    '''
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_checkpoint)
    for k in ['dev', 'test']:
        BABY_LM_DIR = join(BABYLM_ROOT_DIR, f'babylm_{k}')
        files = [join(BABY_LM_DIR, f)
                 for f in os.listdir(BABY_LM_DIR) if f.endswith(f'.{k}')]
        texts = '\n\n'.join([open(f).read() for f in files])
        tokens = []
        chunk_size = 1000
        for i in tqdm(range(0, len(texts), chunk_size)):
            tokens += tokenizer(texts[i:i+chunk_size])['input_ids']
        joblib.dump(tokens, join(BABY_LM_DIR, 'full.joblib'))
