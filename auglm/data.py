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
TOKENS_MINI_GPT2 = [818, 262, 5897, 7404, 286, 33021, 13509, 11, 16343, 992, 1022, 10708, 18639, 290, 48508, 16479, 11, 5615, 281, 1468, 16985, 877, 3706, 25204, 13, 2080, 2832, 356, 8638, 416, 640, 290, 9260, 11, 25204, 19960, 284, 465, 11376, 351, 257, 1842, 523, 2769, 340, 925, 262, 12734, 37854, 290, 262, 7150, 1302, 257, 1643, 25242, 13, 1881, 6076, 3329, 11, 355, 262, 717, 24823, 286, 34488, 41159, 262, 390, 21768, 1633, 11, 25204, 5071, 257, 4071, 4171, 8278, 24924, 3383, 1871, 262, 5417, 286, 30966, 290, 7577, 960, 64, 8278, 339, 550, 2982, 286, 287, 3923, 475, 1239, 4762, 284, 2152,
                    13, 770, 40336, 1064, 2627, 262, 1561, 286, 262, 7404, 11, 8263, 11040, 2951, 290, 11069, 11954, 284, 25204, 338, 11376, 13, 887, 287, 465, 11501, 11, 25204, 2993, 262, 8278, 2492, 470, 4001, 329, 16117, 393, 15807, 26, 340, 373, 257, 15438, 422, 262, 4534, 11, 257, 6194, 286, 262, 8737, 290, 10715, 326, 7363, 287, 262, 24043, 7188, 286, 1204, 13, 679, 19951, 329, 262, 4171, 8278, 11, 9616, 340, 22191, 287, 663, 3288, 1363, 11, 981, 8282, 465, 4445, 25797, 11, 7743, 477, 508, 8672, 326, 262, 45768, 395, 29561, 389, 1690, 7104, 287, 8631, 6504, 11, 23868, 1522, 416, 16336, 290, 1842, 13]


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


if __name__ == '__main__':
    tokenizer_checkpoint = 'gpt2'

    # run this once to preprocess babylm data
    # preprocess_babylm_data(tokenizer_checkpoint=tokenizer_checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_checkpoint)
    dset_mini = NextWordDataset(raw_tokens=TOKENS_MINI_GPT2, max_n_tokens=4)
    tokens, token_next = dset_mini[0]
    print(repr(tokenizer.decode(tokens)), '->',
          repr(tokenizer.decode(token_next)))

    # print('dset len', len(dset_mini))
    # print(dset_mini[len(dset_mini) - 1])
    # # # preprocess
