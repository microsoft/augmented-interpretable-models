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


class HDLM:
    def __init__(
            self, checkpoint,
            emb_size=10000,
            learning_rate=0.1,
            context_length=5,
            device='cuda',
            seed=42,
    ):

        torch.manual_seed(seed)
        np.random.seed(seed)
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        vocab_size = len(self.tokenizer)
        self.learning_rate = learning_rate
        self.vocab = torch.rand((vocab_size, emb_size)).to(device)  # uniform
        # self._normalize_vocab()
        # self.vocab[self.vocab < 0.1] = 0  # binarize with threshold
        # self.vocab[self.vocab >= 0.1] = 1

        self.emb_size = emb_size
        self.context_length = context_length
        self.positional_vectors = torch.Tensor(torchhd.level(
            self.context_length, self.emb_size)).to(device)  # context_length x emb_size
        self.device = device
        self.seed = seed

    def next_emb_from_token_ids(self, input_ids: torch.Tensor):
        '''All the inductive bias comes from this function
        Returns next emb (emb_size)
        '''
        input_ids = input_ids.squeeze()

        # ensure size matches context_length
        diff_inputs_too_long = input_ids.numel() - self.context_length

        # inputs too long (left truncate)
        if input_ids.numel() > self.context_length:
            input_ids = input_ids[-self.context_length:]

        # inputs too short (left pad)
        elif input_ids.numel() < self.context_length:
            input_ids = torch.cat((
                torch.zeros(self.context_length -
                            input_ids.numel(), dtype=torch.int),
                input_ids
            )).unsqueeze(0)

        embs = self.get_embs(input_ids)  # (len(input_ids), emb_size)
        if diff_inputs_too_long < 0:
            embs[:diff_inputs_too_long] = 0

        # multiply with positional vectors (context_length, emb_size) then take mean
        embs = embs * self.positional_vectors  # elementwise multiplication
        return torch.mean(embs, dim=0).squeeze()

    def get_embs(self, input_ids):
        '''Returns array of shape (len(input_ids), emb_size)
        '''
        if isinstance(input_ids, int) or isinstance(input_ids, float):
            input_ids = [input_ids]
        return torch.vstack([self.vocab[i] for i in input_ids])

    def emb_to_token_id(self, emb):
        '''Returns token id
        '''
        probs = self.emb_to_token_probs(emb)
        return torch.argmax(probs)

    def emb_to_token_probs(self, emb):
        '''Returns token probabilities
        '''
        return torch.softmax(self.vocab @ emb, dim=0)
        # return torch.softmax(torch.matmul(self.vocab, emb), dim=0)
        # return torch.softmax(-torch.norm(self.vocab - emb, dim=1), dim=0)

    def update_vocab_emb(self, predicted_emb, next_token_correct_id):
        emb = self.vocab[next_token_correct_id]
        self.vocab[next_token_correct_id] = (
            1 - self.learning_rate) * emb + self.learning_rate * predicted_emb

    def calc_perplexity(self, dset, train=False, n_examples=None, seed=None):
        '''Returns perplexity
        '''
        if n_examples is None:
            n_examples = len(dset)
        if seed is None:
            example_nums = np.arange(n_examples)
        else:
            rng = np.random.default_rng(seed)
            example_nums = rng.choice(
                np.arange(len(dset)), size=n_examples, replace=False)
        log_probs = torch.Tensor(n_examples)
        i = 0
        for ex_num in example_nums:
            prev_token_ids, next_token_id = dset[ex_num]
            emb = self.next_emb_from_token_ids(prev_token_ids)
            probs = self.emb_to_token_probs(emb)
            log_probs[i] = probs[next_token_id].item()

            if train:
                self.update_vocab_emb(emb, next_token_id)

            i += 1

        if train:
            self._normalize_vocab()
        # log_probs = torch.log(log_probs)

        # would normally take torch.exp, but log-perplexity is easier to analyze
        return torch.mean(log_probs).item()

    def _normalize_vocab(self):
        self.vocab /= torch.norm(self.vocab, dim=1).unsqueeze(1)
