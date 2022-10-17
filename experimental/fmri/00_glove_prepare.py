# ! wget https://nlp.stanford.edu/data/glove.840B.300d.zip
# mv glove.840B.300d.zip ~
# cd ~
# unzip glove.840B.300d.zip
# mkdir nlp_utils
# mkdir nlp_utils/glove
# mv glove* nlp_utils/glove/

from tqdm import tqdm
import numpy as np
from os.path import join
import os
# glove_dir = '/home/chansingh/mntv1/nlp_utils/'
glove_dir = '/home/chansingh/nlp_utils/glove'
os.makedirs(glove_dir, exist_ok=True)
fname = join(glove_dir, 'glove.840B.300d.txt')

vocab, embeddings = [], []
print('reading file...')
with open(fname, 'rt') as fi:
    full_content = fi.read().strip().split('\n')

for i in tqdm(range(len(full_content))):
    i_word = full_content[i].split(' ')[0]
    i_embeddings = [float(val) for val in full_content[i].split(' ')[1:]]
    vocab.append(i_word)
    embeddings.append(i_embeddings)

print('converting to np...')
vocab_npa = np.array(vocab)
embs_npa = np.array(embeddings)

# insert '<pad>' and '<unk>' tokens at start of vocab_npa.
vocab_npa = np.insert(vocab_npa, 0, '<pad>')
vocab_npa = np.insert(vocab_npa, 1, '<unk>')
print(vocab_npa[:10])

pad_emb_npa = np.zeros((1, embs_npa.shape[1]))  # embedding for '<pad>' token.
# embedding for '<unk>' token.
unk_emb_npa = np.mean(embs_npa, axis=0, keepdims=True)

# insert embeddings for pad and unk tokens at top of embs_npa.
embs_npa = np.vstack((pad_emb_npa, unk_emb_npa, embs_npa))
print(embs_npa.shape)

print('saving...')
with open(join(glove_dir, 'vocab_npa.npy'), 'wb') as f:
    np.save(f, vocab_npa)

with open(join(glove_dir, 'embs_npa.npy'), 'wb') as f:
    np.save(f, embs_npa)
