import numpy as np
import itertools as itools
from .DataSequence import DataSequence

DEFAULT_BAD_WORDS = frozenset(["sentence_start", "sentence_end", "br", "lg", "ls", "ns", "sp"])

def make_word_ds(grids, trfiles, bad_words=DEFAULT_BAD_WORDS):
    """Creates DataSequence objects containing the words from each grid, with any words appearing
    in the [bad_words] set removed.
    """
    ds = dict()
    stories = list(set(trfiles.keys()) & set(grids.keys()))
    for st in stories:
        grtranscript = grids[st].tiers[1].make_simple_transcript()
        ## Filter out bad words
        goodtranscript = [x for x in grtranscript
                          if x[2].lower().strip("{}").strip() not in bad_words]
        d = DataSequence.from_grid(goodtranscript, trfiles[st][0])
        ds[st] = d

    return ds

def make_phoneme_ds(grids, trfiles):
    """Creates DataSequence objects containing the phonemes from each grid.
    """
    ds = dict()
    stories = grids.keys()
    for st in stories:
        grtranscript = grids[st].tiers[0].make_simple_transcript()
        d = DataSequence.from_grid(grtranscript, trfiles[st][0])
        ds[st] = d

    return ds

phonemes = ['AA', 'AE','AH','AO','AW','AY','B','CH','D', 'DH', 'EH', 'ER', 'EY', 
            'F', 'G', 'HH', 'IH', 'IY', 'JH','K', 'L', 'M', 'N', 'NG', 'OW', 'OY', 
            'P', 'R', 'S', 'SH', 'T', 'TH', 'UH', 'UW', 'V', 'W', 'Y', 'Z', 'ZH']

def make_character_ds(grids, trfiles):
    ds = dict()
    stories = grids.keys()
    for st in stories:
        grtranscript = grids[st].tiers[2].make_simple_transcript()
        fixed_grtranscript = [(s,e,map(int, c.split(","))) for s,e,c in grtranscript if c]
        d = DataSequence.from_grid(fixed_grtranscript, trfiles[st][0])
        ds[st] = d
    return ds

def make_dialogue_ds(grids, trfiles):
    ds = dict()
    for st, gr in grids.iteritems():
        grtranscript = gr.tiers[3].make_simple_transcript()
        fixed_grtranscript = [(s,e,c) for s,e,c in grtranscript if c]
        ds[st] = DataSequence.from_grid(fixed_grtranscript, trfiles[st][0])
    return ds

def histogram_phonemes(ds, phonemeset=phonemes):
    """Histograms the phonemes in the DataSequence [ds].
    """
    olddata = ds.data
    N = len(ds.data)
    newdata = np.zeros((N, len(phonemeset)))
    phind = dict(enumerate(phonemeset))
    for ii,ph in enumerate(olddata):
        try:
            #ind = phonemeset.index(ph.upper().strip("0123456789"))
            ind = phind[ph.upper().strip("0123456789")]
            newdata[ii][ind] = 1
        except Exception as e:
            pass

    return DataSequence(newdata, ds.split_inds, ds.data_times, ds.tr_times)

def histogram_phonemes2(ds, phonemeset=phonemes):
    """Histograms the phonemes in the DataSequence [ds].
    """
    olddata = np.array([ph.upper().strip("0123456789") for ph in ds.data])
    newdata = np.vstack([olddata==ph for ph in phonemeset]).T
    return DataSequence(newdata, ds.split_inds, ds.data_times, ds.tr_times)

def apply_model_to_words(ds: DataSequence, lsa_semantic_model, size: int) -> DataSequence:
    """
    ds
        datasequence to operate on
    lsa_semantic_model
        semantic model to use
    size
        size of resulting vectors from the semantic model
    """
    newdata = []
    for w in ds.data:
        v = []
        try:
            v = np.concatenate((v, lsa_semantic_model[str.encode(w.lower())]))
        except KeyError as e:
            v = np.concatenate((v, np.zeros((size)))) #lsasm.data.shape[0],))
        newdata.append(v)
    return DataSequence(np.array(newdata), ds.split_inds, ds.data_times, ds.tr_times)

def apply_model_to_ngrams(ds: DataSequence, embedding_function, ngram_size: int=5) -> DataSequence:
    """
    ds
        datasequence to operate on
    embedding_function
        ngram -> fixed size vector
    size
        size of resulting vectors from the embedding function model
    """
    ngrams_list = []
    for i in range(len(ds.data)):
        l = max(0, i - ngram_size)
        ngram = ' '.join(ds.data[l: i + 1])
        ngrams_list.append(ngram)

    # out_list is (batch_size, 1, (seq_len + 2), 768) -- BERT adds initial / final tokens
    out_list = embedding_function(ngrams_list)

    # convert to np array by averaging over len (can't just convert this since seq lens vary)
    # embs = np.array(out).squeeze().mean(axis=1)
    num_ngrams = len(out_list)
    dim_size = len(out_list[0][0][0])
    embs = np.zeros((num_ngrams, dim_size))
    return DataSequence(embs, ds.split_inds, ds.data_times, ds.tr_times)


def make_character_model(dss):
    """Make character indicator model for a dict of datasequences.
    """
    stories = dss.keys()
    storychars = dict([(st,np.unique(np.hstack(ds.data))) for st,ds in dss.iteritems()])
    total_chars = sum(map(len, storychars.values()))
    char_inds = dict()
    ncharsdone = 0
    for st in stories:
        char_inds[st] = dict(zip(storychars[st], range(ncharsdone, ncharsdone+len(storychars[st]))))
        ncharsdone += len(storychars[st])

    charmodels = dict()
    for st,ds in dss.iteritems():
        charmat = np.zeros((len(ds.data), total_chars))
        for ti,charlist in enumerate(ds.data):
            for char in charlist:
                charmat[ti, char_inds[st][char]] = 1
        charmodels[st] = DataSequence(charmat, ds.split_inds, ds.data_times, ds.tr_times)

    return charmodels, char_inds

def make_dialogue_model(ds):
    return DataSequence(np.ones((len(ds.data),1)), ds.split_inds, ds.data_times, ds.tr_times)

def modulate(ds, vec):
    """Multiplies each row (each word/phoneme) by the corresponding value in [vec].
    """
    return DataSequence((ds.data.T*vec).T, ds.split_inds, ds.data_times, ds.tr_times)

def catmats(*seqs):
    keys = seqs[0].keys()
    return dict([(k, DataSequence(np.hstack([s[k].data for s in seqs]), seqs[0][k].split_inds)) for k in keys])
