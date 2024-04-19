import warnings
import re
from tqdm import tqdm
import torch
from os.path import join
import transformers
import pandas as pd
import transformers
import scipy
import os
import datasets
import sys
from copy import deepcopy
import numpy as np


def get_next_word_distr(
    prefix: str,
    model: transformers.PreTrainedModel,
    tokenizer: transformers.PreTrainedTokenizer,
    num_tokens: int = 2,
    cdf_threshold: float = 0.8,
    batch_size: int = 2048,
) -> pd.DataFrame:
    '''Given an LM, get the distribution of the next word. Since LMs use tokens, this can vary from the next-token distr.
    To solve this issue, we compute the prob distr. for 2 or 3 tokens, then merge the probabilities that yield the same leading word.

    Note: to compute the leading word, we convert text to lowercase, trim (remove whitespace), and remove punctuation.

    Params
    ------
    prefix
        The string preceding the next-word distr (generally best if this does not end with a space)
        Example: 'My favorite ice cream flavor is'
    model
    tokenizer
    num_tokens
        2 or 3. How many tokens do we use to compute the distr?
    cdf_threshold (0-1)
        Instead of computing the entire distribution, we can compute the distribution of the top cdf_threshold fraction of the probability mass at each token
        If we sample 2 tokens, we will then have the top cdf_threshold probability mass
        For 3 tokens, we will have the top cdf_threshold^2 probability mass

    Returns
    -------
    next_word_distr: pd.DataFrame
        A dataframe with columns 'generation' and 'prob'.
        The generation is the next word, and the prob is the probability of that word.
    '''

    # unigram
    voc_size = len(tokenizer)
    input_ids_tok0 = tokenizer.encode(
        prefix, return_tensors="pt")  # .to(model.device)
    tok0_logits = model(
        input_ids=input_ids_tok0).logits[:, -1, :].detach().cpu().numpy().astype(np.double)
    tok0_probs = scipy.special.softmax(tok0_logits).flatten()

    # cdf of unigram1_probs
    tok0_argsort = np.argsort(tok0_probs)[::-1]
    tok0_cdf = np.cumsum(tok0_probs[tok0_argsort])
    num_selected_tok0 = np.argmax(tok0_cdf > cdf_threshold)
    # ngram_probs.append(tok0_probs[tok0_argsort[:num_selected_tok0]])
    tok0_idxs = tok0_argsort[:num_selected_tok0]
    tok0_ids = tok0_idxs  # token ids
    print('num_tokens_selected', num_selected_tok0)
    if num_tokens == 1:
        generations = [tokenizer.decode([tok_id]) for tok_id in tok0_ids]
        return _convert_to_word_distr(generations, tok0_probs[tok0_idxs])

    # 2 tokens ##########################
    # compute ngram probs for the next ngram
    prefix_embs = tokenizer.encode(prefix, return_tensors="pt")[0]
    input_ids_tok1 = torch.stack([
        torch.concatenate(
            (prefix_embs, torch.LongTensor([tok_id])))
        for tok_id in tok0_ids
    ])

    tok1_logits = torch.zeros((num_selected_tok0, voc_size))
    for i in tqdm(range(0, num_selected_tok0, batch_size)):
        input_ids = input_ids_tok1[i:i + batch_size]
        tok1_logits[i:i + batch_size] = \
            model(input_ids).logits[:, -1, :].detach().cpu()
    tok1_logits = tok1_logits.numpy().astype(np.double)

    # this has prob for each token given the prefix
    print('softmax...')
    tok1_probs_conditional = scipy.special.softmax(tok1_logits, axis=1)

    # multiply by prob of tok0 to get raw probs
    tok1_probs = tok1_probs_conditional * tok0_probs[tok0_idxs].reshape(-1, 1)

    assert np.allclose(tok1_probs.sum(axis=1),
                       tok0_probs[tok0_idxs], atol=1e-6)
    assert np.allclose(tok1_probs.sum(), cdf_threshold, atol=1e-1)

    if num_tokens > 2:
        # get top token idxs
        tok1_argsort = np.argsort(tok1_probs.flatten())[::-1]
        tok1_cdf = np.cumsum(tok1_probs.flatten()[tok1_argsort])
        num_selected_tok1 = np.argmax(tok1_cdf > cdf_threshold * cdf_threshold)
        tok1_selected_idxs = tok1_argsort[:num_selected_tok1]

        tok1_idxs_selected_rows, tok1_idxs_selected_cols = np.unravel_index(
            tok1_selected_idxs, tok1_probs.shape)
        tok0_ids_1 = tok0_idxs[tok1_idxs_selected_rows]
        tok1_ids_1 = tok1_idxs_selected_cols
        probs_ids_1 = tok1_probs[tok1_idxs_selected_rows,
                                 tok1_idxs_selected_cols]
    else:
        # repeat tok0_ids_1 voc_size times
        tok0_ids_1 = np.repeat(tok0_idxs, voc_size)
        tok1_ids_1 = np.tile(np.arange(voc_size), num_selected_tok0)
        probs_ids_1 = tok1_probs.flatten()

    # convert to row, col idxs and recreate tokens
    print('unraveling...')

    if num_tokens == 2:
        tokens_list = [[tok0_ids_1[i], tok1_ids_1[i]]
                       for i in range(len(tok0_ids_1))]
        # generations = tokenizer.batch_decode(tokens_list)
        # decode in batches
        batch_size = 4096
        generations = []
        for i in tqdm(range(0, len(tokens_list), batch_size)):
            generations[i: i + batch_size] = tokenizer.batch_decode(
                tokens_list[i: i + batch_size])

        print('convert to word distr...')
        return _convert_to_word_distr(generations, probs_ids_1)

    # 3 tokens ##########################
    # repeat for tok2
    input_ids_tok2 = torch.stack([
        torch.concatenate(
            (prefix_embs, torch.LongTensor([tok0_ids_1[i]]), torch.LongTensor([tok1_ids_1[i]])))
        for i in range(num_selected_tok1)
    ])
    print('num_tokens2', num_selected_tok1)
    tok2_logits = torch.zeros((num_selected_tok1, voc_size))
    for i in tqdm(range(0, num_selected_tok1, batch_size)):
        input_ids = input_ids_tok2[i:i + batch_size]
        tok2_logits[i:i + batch_size] = \
            model(input_ids).logits[:, -1, :].detach().cpu()
    tok2_logits = tok2_logits.numpy().astype(np.double)
    tok2_probs_conditional = scipy.special.softmax(tok2_logits, axis=1)
    tok2_probs = tok2_probs_conditional * probs_ids_1.reshape(-1, 1)

    # get top token idxs
    if num_tokens > 3:
        tok2_argsort = np.argsort(tok2_probs.flatten())[::-1]
        tok2_cdf = np.cumsum(tok2_probs.flatten()[tok2_argsort])
        num_selected_tok2 = np.argmax(
            tok2_cdf > cdf_threshold * cdf_threshold * cdf_threshold)
        tok2_selected_idxs = tok2_argsort[:num_selected_tok2]
    else:
        tok2_selected_idxs = np.arange(tok2_probs.shape[1])

    # convert to row, col idxs and recreate tokens
    tok2_idxs_selected_rows, tok2_idxs_selected_cols = np.unravel_index(
        tok2_selected_idxs, tok2_probs.shape)
    tok0_ids_2 = tok0_ids_1[tok2_idxs_selected_rows]
    tok1_ids_2 = tok1_ids_1[tok2_idxs_selected_rows]
    tok2_ids_2 = tok2_idxs_selected_cols
    probs_ids_2 = tok2_probs[tok2_idxs_selected_rows, tok2_idxs_selected_cols]

    generations = [tokenizer.decode([tok0_ids_2[i], tok1_ids_2[i], tok2_ids_2[i]])
                   for i in range(len(tok0_ids_2))]
    return _convert_to_word_distr(generations, probs_ids_2)


def _convert_to_word_distr(generations, probs) -> pd.DataFrame:
    df = pd.DataFrame({
        'generation': generations,
        'prob': probs
    })

    def remove_punctuation(s):
        return re.sub(r'[^\w\s]', '', s)

    def is_whitespace(s):
        return re.match(r'^\s*$', s) is not None

    def get_first_word(s):
        # strip lowercase and remove punctuation
        s = s.strip().lower()
        s = remove_punctuation(s)
        words = s.split()
        if len(words) == 0:
            return ' '
        return words[0]

    df['generation'] = df['generation'].apply(get_first_word)

    # remove any rows that are just whitespace
    idxs_whitespace = df['generation'].apply(is_whitespace)
    if idxs_whitespace.sum() > 0:
        warnings.warn(
            f'removing {idxs_whitespace.sum()} whitespace rows accounting for {df[idxs_whitespace]["prob"].sum()} prob')
    df = df[~df['generation'].apply(is_whitespace)]

    # merge prob based on first_word
    df = df.groupby('generation')['prob'].sum(
    ).reset_index().sort_values(by='prob', ascending=False)
    return df


if __name__ == '__main__':
    import imodelsx.llm

    # these are just normal HF models/tokenizers
    checkpoint = 'gpt2-xl'
    model = imodelsx.llm.load_hf_model(checkpoint)
    tokenizer = imodelsx.llm.load_tokenizer(checkpoint)

    d = get_next_word_distr(
        prefix='My favorite ice cream flavor is',
        model=model,
        tokenizer=tokenizer,
        num_tokens=2,
        cdf_threshold=0.7,
        batch_size=2048,
    )

    display(d)
