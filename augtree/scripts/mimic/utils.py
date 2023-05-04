import os
import pickle
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from scipy.sparse._csr import csr_matrix
from tqdm import tqdm


def load_mimic_data_and_vocab(path: str):
    with open(os.path.join(path, "mimiciv-2.2.pkl"), "rb") as f:
        data = pickle.load(f)

    vocab_path = os.path.join(path, "mimiciv-2.2-vocab.pkl")
    if os.path.exists(vocab_path):
        with open(vocab_path, "rb") as f:
            vocab = pickle.load(f)
    else:
        vocab = get_mimic_vocab(data)
        with open(vocab_path, "wb") as f:
            pickle.dump(vocab, f)

    return data, vocab


def get_unique_words(list_of_phrases: list[str]):
    unique_words = set()
    for phrase in list_of_phrases:
        unique_words = unique_words.union(set(phrase.lower().split(" ")))
    return unique_words


def get_mimic_vocab(data: dict):
    vocab = set()
    for key in ["diagnoses", "procedures", "discharge_notes"]:
        print(f"Getting vocab for {key}")
        with ThreadPoolExecutor(max_workers=64) as executor:
            futures = []
            for list_of_phrases in tqdm(data[key]):
                futures.append(executor.submit(get_unique_words, list_of_phrases))
            key_vocab = set()
            for future in futures:
                words = future.result()
                key_vocab = key_vocab.union(set(words))
        vocab = vocab.union(key_vocab)
    return list(vocab)


def get_mimic_X_y(data: dict, label="mortality"):
    """
    get the text data and labels for training

    Args:
        data (dict): has keys ['mortality', 'readmission', 'diagnoses', 'procedures', 'discharge_notes']
        label (str): "mortality" or "readmission" target used to generate labels
    Returns:
        X (list): list of text data
        y (list): binary target
    """
    assert label in ["mortality", "readmission"], "label type not understood"
    assert (
        len(set([len(data[x]) for x in data.keys()])) == 1
    ), "lengths of dict keys dont match"
    n = len(data["diagnoses"])

    # Concatenate all the strings together into one big string
    X = [
        (
            "diagnosis "
            + " diagnosis ".join(data["diagnoses"][i])
            + "procedures "
            + " procedures ".join(data["procedures"][i])
            + "".join(data["discharge_notes"][i].replace("\n", " "))
        ).lower()
        for i in range(n)
    ]
    y = data[label]
    return X, y


def split_mimic_data(
    X_counts: csr_matrix, X: list, y: list, test_size: float, subsample_frac: float = 1
):
    """
    split the data into training and test sets based on size subsample_frac

    Args:
        X_counts (csr_matrix): (i,j) entry is how many times word j appeared for data point i
        X (list): list of all text data (diagnoses/procedure descriptions and mimic notes)
        y (list): binary target
        test_size (float): fraction of data to be used for test set
        subsample_frac (float): fraction of data to be used for training
    Returns:
        X_train_counts (csr_matrix)
        X_train_text (list)
        X_test_counts (csr_matrix)
        X_test_text (list)
        y_train (list)
        y_test (list)
    """

    assert subsample_frac >= 0 and subsample_frac <= 1, "subsample frac not in [0,1]"
    assert test_size >= 0 and test_size <= 1, "test size not in [0,1]"
    assert len(X) == len(y), "size of data and labels dont match"

    if subsample_frac < 1:
        n = len(y)
        idx_keep = np.random.choice(
            list(range(n)), int(subsample_frac * n), replace=False
        )
        X_counts = [X_counts[i] for i in idx_keep]
        X = [X[i] for i in idx_keep]
        y = [y[i] for i in idx_keep]

    # split into train and test indices
    n = len(y)
    idxs = list(range(n))
    idx_train = np.random.choice(idxs, int(test_size * n), replace=False)
    idx_test = np.array(list(set(idxs).difference(set(idx_train))))

    X_train_counts = X_counts[idx_train, :]
    X_train_text = [X[i] for i in idx_train]
    X_test_counts = X_counts[idx_test, :]
    X_test_text = [X[i] for i in idx_test]
    y_train = [y[i] for i in idx_train]
    y_test = [y[i] for i in idx_test]

    return X_train_counts, X_train_text, X_test_counts, X_test_text, y_train, y_test
