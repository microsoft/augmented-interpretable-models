import os
import sys
import numpy as np
import json
from os.path import join, dirname
from functools import partial
from ridge_utils.DataSequence import DataSequence
from typing import Dict
from tqdm import tqdm
from ridge_utils.interpdata import lanczosinterp2D
from ridge_utils.SemanticModel import SemanticModel
from ridge_utils.dsutils import apply_model_to_ngrams, apply_model_to_words, make_word_ds, make_phoneme_ds
from ridge_utils.stimulus_utils import load_textgrids, load_simulated_trfiles
from transformers import pipeline

repo_dir = '/home/chansingh/mntv1/deep-fMRI' # join(dirname(dirname(os.path.abspath(__file__))))
nlp_utils_dir = '/home/chansingh/nlp_utils'
em_data_dir = join(repo_dir, 'em_data')
data_dir = join(repo_dir, 'data')
results_dir = join(repo_dir, 'results')

def get_story_wordseqs(stories) -> Dict[str, DataSequence]:
	grids = load_textgrids(stories, data_dir)
	with open(join(data_dir, "ds003020/derivative/respdict.json"), "r") as f:
		respdict = json.load(f)
	trfiles = load_simulated_trfiles(respdict)
	wordseqs = make_word_ds(grids, trfiles)
	return wordseqs

def get_story_phonseqs(stories):
	grids = load_textgrids(stories, data_dir)
	with open(join(data_dir, "ds003020/derivative/respdict.json"), "r") as f:
		respdict = json.load(f)
	trfiles = load_simulated_trfiles(respdict)
	wordseqs = make_phoneme_ds(grids, trfiles)
	return wordseqs

def downsample_word_vectors(stories, word_vectors, wordseqs):
	"""Get Lanczos downsampled word_vectors for specified stories.

	Args:
		stories: List of stories to obtain vectors for.
		word_vectors: Dictionary of {story: <float32>[num_story_words, vector_size]}

	Returns:
		Dictionary of {story: downsampled vectors}
	"""
	downsampled_semanticseqs = dict()
	for story in stories:
		downsampled_semanticseqs[story] = lanczosinterp2D(
			word_vectors[story], wordseqs[story].data_times, 
			wordseqs[story].tr_times, window=3)
	return downsampled_semanticseqs

def ph_to_articulate(ds, ph_2_art):
	""" Following make_phoneme_ds converts the phoneme DataSequence object to an 
	articulate Datasequence for each grid.
	"""
	articulate_ds = []
	for ph in ds:
		try:
			articulate_ds.append(ph_2_art[ph])
		except:
			articulate_ds.append([""])
	return articulate_ds

articulates = ["bilabial","postalveolar","alveolar","dental","labiodental",
			   "velar","glottal","palatal", "plosive","affricative","fricative",
			   "nasal","lateral","approximant","voiced","unvoiced","low", "mid",
			   "high","front","central","back"]

def histogram_articulates(ds, data, articulateset=articulates):
	"""Histograms the articulates in the DataSequence [ds]."""
	final_data = []
	for art in ds:
		final_data.append(np.isin(articulateset, art))
	final_data = np.array(final_data)
	return (final_data, data.split_inds, data.data_times, data.tr_times)

def get_articulation_vectors(allstories):
	"""Get downsampled articulation vectors for specified stories.
	Args:
		allstories: List of stories to obtain vectors for.
	Returns:
		Dictionary of {story: downsampled vectors}
	"""
	with open(join(em_data_dir, "articulationdict.json"), "r") as f:
		artdict = json.load(f)
	phonseqs = get_story_phonseqs(allstories) #(phonemes, phoneme_times, tr_times)
	downsampled_arthistseqs = {}
	for story in allstories:
		olddata = np.array(
			[ph.upper().strip("0123456789") for ph in phonseqs[story].data])
		ph_2_art = ph_to_articulate(olddata, artdict)
		arthistseq = histogram_articulates(ph_2_art, phonseqs[story])
		downsampled_arthistseqs[story] = lanczosinterp2D(
			arthistseq[0], arthistseq[2], arthistseq[3])
	return downsampled_arthistseqs


def get_phonemerate_vectors(allstories):
	"""Get downsampled phonemerate vectors for specified stories.
	Args:
		allstories: List of stories to obtain vectors for.
	Returns:
		Dictionary of {story: downsampled vectors}
	"""
	with open(join(em_data_dir, "articulationdict.json"), "r") as f:
		artdict = json.load(f)
	phonseqs = get_story_phonseqs(allstories) #(phonemes, phoneme_times, tr_times)
	downsampled_arthistseqs = {}
	for story in allstories:
		olddata = np.array(
			[ph.upper().strip("0123456789") for ph in phonseqs[story].data])
		ph_2_art = ph_to_articulate(olddata, artdict)
		arthistseq = histogram_articulates(ph_2_art, phonseqs[story])
		nphonemes = arthistseq[0].shape[0]
		phonemerate = np.ones([nphonemes, 1])
		downsampled_arthistseqs[story] = lanczosinterp2D(
			phonemerate, arthistseq[2], arthistseq[3])
	return downsampled_arthistseqs

def get_wordrate_vectors(allstories):
	"""Get wordrate vectors for specified stories.

	Args:
		allstories: List of stories to obtain vectors for.

	Returns:
		Dictionary of {story: downsampled vectors}
	"""
	eng1000 = SemanticModel.load(join(em_data_dir, "english1000sm.hf5"))
	wordseqs = get_story_wordseqs(allstories)
	vectors = {}
	for story in allstories:
		nwords = len(wordseqs[story].data)
		vectors[story] = np.ones([nwords, 1])
	return downsample_word_vectors(allstories, vectors, wordseqs)


def get_eng1000_vectors(allstories):
	"""Get Eng1000 vectors (985-d) for specified stories.

	Args:
		allstories: List of stories to obtain vectors for.

	Returns:
		Dictionary of {story: downsampled vectors}
	"""
	eng1000 = SemanticModel.load(join(em_data_dir, "english1000sm.hf5"))
	wordseqs = get_story_wordseqs(allstories)
	vectors = {}
	for story in allstories:
		sm = apply_model_to_words(wordseqs[story], eng1000, 985)
		vectors[story] = sm.data
		print(sm.data.shape)
	return downsample_word_vectors(allstories, vectors, wordseqs)

def get_glove_vectors(allstories):
	"""Get glove vectors (300-d) for specified stories.

	Args:
		allstories: List of stories to obtain vectors for.

	Returns:
		Dictionary of {story: downsampled vectors}
	"""
	glove = SemanticModel.load_np(join(nlp_utils_dir, 'glove'))
	wordseqs = get_story_wordseqs(allstories)
	vectors = {}
	for story in allstories:
		sm = apply_model_to_words(wordseqs[story], glove, 300)
		vectors[story] = sm.data
	return downsample_word_vectors(allstories, vectors, wordseqs)

def get_bert_vectors(allstories, model='bert-base-uncased', ngram_size=5):
	"""Get bert vectors
	"""
	pipe = pipeline("feature-extraction", model=model, device=0)
	wordseqs = get_story_wordseqs(allstories)
	vectors = {}
	
	print('extracting bert vecs...')
	for story in tqdm(allstories):
		sm = apply_model_to_ngrams(wordseqs[story], pipe, ngram_size=ngram_size)
		vectors[story] = sm.data
	return downsample_word_vectors(allstories, vectors, wordseqs)



############################################
########## Feature Space Creation ##########
############################################

_FEATURE_CONFIG = {
	"articulation": get_articulation_vectors,
	"phonemerate": get_phonemerate_vectors,
	"wordrate": get_wordrate_vectors,
	"eng1000": get_eng1000_vectors,
	'glove': get_glove_vectors,
	'bert-3': partial(get_bert_vectors, ngram_size=3),
	'bert-5': partial(get_bert_vectors, ngram_size=5),
	'bert-10': partial(get_bert_vectors, ngram_size=10),
	'bert-20': partial(get_bert_vectors, ngram_size=20),
}

def get_feature_space(feature, *args):
	return _FEATURE_CONFIG[feature](*args)

if __name__ == '__main__':
	feats = get_feature_space('bert-5', ['sloth'])
