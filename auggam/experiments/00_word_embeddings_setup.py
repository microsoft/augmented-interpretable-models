from gensim.scripts.glove2word2vec import glove2word2vec
import gensim
from os.path import join
from gensim.models import KeyedVectors

def download_setup():
    # Global parameters
    # root folder
    root_folder = '/home/chansingh/nlp_utils/glove/'
    glove_filename = 'glove.840B.300d.txt'

    # Variable for data directory
    glove_path = join(root_folder, glove_filename)

    #glove_input_file = glove_filename
    word2vec_output_file = glove_filename + '.word2vec'
    glove2word2vec(glove_path, word2vec_output_file)

    # word2vec setup
    import gensim.downloader as api
    wv = api.load('word2vec-google-news-300')

    # move them both into the write folder....

# m = gensim.models.Word2Vec.load('/home/chansingh/nlp_utils/glove/glove.840B.300d.txt.word2vec')
# m = gensim.models.Word2Vec.load('/home/chansingh/nlp_utils/word2vec/word2vec-google-news-300.wordvectors.vectors.npy', mmap='r')
# wv_from_text = KeyedVectors.load_word2vec_format(datapath('word2vec_pre_kv_c'), binary=False)
wv = KeyedVectors.load("/home/chansingh/nlp_utils/word2vec/word2vec-google-news-300.wordvectors.vectors.npy", mmap='r')