from imodelsx import EmbGAMClassifier
import pandas as pd
import numpy as np
import sklearn.metrics
import pickle as pkl
import embgam.embed
import warnings
from sklearn.base import ClassifierMixin, RegressorMixin


def sparse_predict(model, X, warn, frac_ngrams=1):
    """Predict only the cached coefs in self.coefs_dict_"""
    assert hasattr(model, "coefs_dict_"), "coefs are not cached!"
    preds = []
    n_unseen_ngrams = 0
    for x in X:
        pred_list = []
        seqs = embgam.embed.generate_ngrams_list(
            x,
            ngrams=model.ngrams,
            tokenizer_ngrams=model.tokenizer_ngrams,
            all_ngrams=model.all_ngrams,
        )

        for seq in seqs:
            if seq in model.coefs_dict_:
                pred_list.append(model.coefs_dict_[seq])
            else:
                n_unseen_ngrams += 1

        def key(x):
            if type(x) is np.ndarray:
                return sum(abs(x))
            return abs(x)

        pred_arr = np.array(sorted(pred_list, key=key, reverse=True))

        if len(pred_arr) == 0:
            preds.append(np.zeros(len(m.classes_)))
        elif frac_ngrams < 1:
            # only look at a fraction of seqs
            if pred_arr.ndim == 1:
                cumsum = np.cumsum(np.abs(pred_arr))
            elif pred_arr.ndim == 2:
                cumsum = np.cumsum(np.sum(np.abs(pred_arr), axis=1))

            # note cumsum[-1] is the total sum of absolute value of all the elements
            idx = np.where((cumsum / cumsum[-1]) > frac_ngrams)[0][0]
            preds.append(sum(pred_arr[: idx + 1]))
        else:
            preds.append(sum(pred_arr))

    if n_unseen_ngrams > 0 and warn:
        warnings.warn(
            f"Saw an unseen ungram {n_unseen_ngrams} times. \
For better performance, call cache_linear_coefs on the test dataset \
before calling predict."
        )

    preds = np.array(preds)
    if isinstance(model, RegressorMixin):
        return preds
    elif isinstance(model, ClassifierMixin):
        if preds.ndim > 1:
            return np.argmax(preds, axis=1)
        return ((preds + model.linear.intercept_) > 0).astype(int)


def predict(m, X, warn):
    """Predict only the cached coefs in self.coefs_dict_"""
    assert hasattr(m, "coefs_dict_"), "coefs are not cached!"
    preds = []
    n_unseen_ngrams = 0
    n_classes = len(m.classes_)
    for x in X:
        if n_classes > 2:
            pred = np.zeros(n_classes)
        else:
            pred = 0
        seqs = embgam.embed.generate_ngrams_list(
            x,
            ngrams=m.ngrams,
            tokenizer_ngrams=m.tokenizer_ngrams,
            all_ngrams=m.all_ngrams,
        )
        for seq in seqs:
            if seq in m.coefs_dict_:
                pred += m.coefs_dict_[seq]
            else:
                n_unseen_ngrams += 1
        preds.append(pred)
    if n_unseen_ngrams > 0 and warn:
        warnings.warn(
            f"Saw an unseen ungram {n_unseen_ngrams} times. \
For better performance, call cache_linear_coefs on the test dataset \
before calling predict."
        )

    preds = np.array(preds)
    if isinstance(m, RegressorMixin):
        return preds
    elif isinstance(m, ClassifierMixin):
        if preds.ndim > 1:  # multiclass classification
            return np.argmax(preds, axis=1)
        else:
            return (preds + m.linear.intercept_ > 0).astype(int)


def get_metrics(m, data, sparse=False, frac_ngrams=-1):
    try:
        text = data["headline"].tolist()
    except AttributeError:
        text = data["headline"]
    label = data["category"]
    # preds = m.predict(text, warn=False)
    if sparse:
        assert frac_ngrams >= 0 and frac_ngrams <= 1, "Need to specify frac_ngrams"
        preds = sparse_predict(m, text, warn=False, frac_ngrams=frac_ngrams)
    else:
        preds = predict(m, text, warn=False)
    acc = sklearn.metrics.accuracy_score(label, preds)
    return acc


# load model
with open(f"Data/huffpost.pkl", "rb") as f:
    huffpost_data = pkl.load(f)

# the keys 0, 1, 2 represent indistribution (ID) training and validation and out of distribution test set
# see table 6 for more info: https://arxiv.org/pdf/2211.14238.pdf
for year in [2012, 2013, 2014, 2015, 2016, 2017, 2018]:
    for key in [0, 1, 2]:
        if type(huffpost_data[year][key]["headline"]) != list:
            huffpost_data[year][key]["headline"] = huffpost_data[year][key][
                "headline"
            ].tolist()


ngrams = 2
for year in [2012]:
    for ngrams in [1, 2, 3, 4]:
        with open(f"models/huffpost_{year}_embgam_ngrams={ngrams}.pkl", "rb") as f:
            m = pkl.load(f)

        dset_train = huffpost_data[year][0]
        dset_val = huffpost_data[year][1]
        dset_test = huffpost_data[year + 1][2]

        print("Year=", year, "ngrams=", ngrams)
        acc = get_metrics(m, dset_train)
        print(f"Train accuracy {acc:0.2f}")
        acc = get_metrics(m, dset_val)
        print(f"Val accuracy {acc:0.2f}")
        acc = get_metrics(m, dset_test)
        print(f"Test accuracy {acc:0.2f}")
        print("Sparse predict Year=", year)
        acc = get_metrics(m, dset_train)
        print(f"Train accuracy {acc:0.2f}")
        acc = get_metrics(m, dset_val, sparse=True, frac_ngrams=0.3)
        print(f"Val accuracy {acc:0.2f}")
        acc = get_metrics(m, dset_test, sparse=True, frac_ngrams=0.3)
        print(f"Test accuracy {acc:0.2f}")
        print("*" * 50)
