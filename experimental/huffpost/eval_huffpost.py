from imodelsx import EmbGAMClassifier
import pandas as pd
import numpy as np
import sklearn.metrics
import pickle as pkl
import embgam.embed
import warnings
from sklearn.base import ClassifierMixin, RegressorMixin


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


def get_metrics(m, data):
    try:
        text = data["headline"].tolist()
    except AttributeError:
        text = data["headline"]
    label = data["category"]
    # preds = m.predict(text, warn=False)
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
for year in [2012, 2013, 2014, 2015, 2016]:
    with open(f"models/huffpost_{year}_embgam_ngrams={ngrams}.pkl", "rb") as f:
        m = pkl.load(f)

    dset_train = huffpost_data[year][0]
    dset_val = huffpost_data[year][1]
    dset_test = huffpost_data[year + 2][2]

    print("Year=", year, "ngrams=", ngrams)
    acc = get_metrics(m, dset_train)
    print(f"Train accuracy {acc:0.2f}")
    acc = get_metrics(m, dset_val)
    print(f"Val accuracy {acc:0.2f}")
    acc = get_metrics(m, dset_test)
    print(f"Test accuracy {acc:0.2f}")
