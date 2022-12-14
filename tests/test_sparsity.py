import numpy as np
import embgam
from sklearn.base import ClassifierMixin, RegressorMixin
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
from sklearn.decomposition import TruncatedSVD


from multiprocessing import Pool
import transformers

from functools import partial

from embgam import data
import pickle as pkl
import warnings

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
dsets = {
    # "financial_phrasebank": "ahmedrachid/FinancialBERT-Sentiment-Analysis", # 3 class classification
    "sst2": "textattack/bert-base-uncased-SST-2",
    # "emotion": "nateraw/bert-base-uncased-emotion", # 6 class classification
    "rotten_tomatoes": "textattack/bert-base-uncased-rotten_tomatoes",
}


def compute_emb_ngram(ngram, model, transformer_model, tokenizer_embeddings):
    tokens = tokenizer_embeddings(
        [ngram], padding=True, truncation=True, return_tensors="pt"
    )
    tokens = tokens.to(transformer_model.device)
    output = transformer_model(**tokens)
    emb = output[model.layer].cpu().detach().numpy()
    if len(emb.shape) == 3:  # includes seq_len
        emb = emb.mean(axis=1)
    return emb


def compute_emb(x, model, transformer_model, tokenizer_embeddings):
    embs = []
    seqs = embgam.embed.generate_ngrams_list(
        x,
        ngrams=model.ngrams,
        tokenizer_ngrams=model.tokenizer_ngrams,
        all_ngrams=model.all_ngrams,
    )

    process_item_with_args = partial(
        compute_emb_ngram,
        model=model,
        transformer_model=transformer_model,
        tokenizer_embeddings=tokenizer_embeddings,
    )
    with Pool(8) as pool:
        embs = list(tqdm(pool.imap(process_item_with_args, seqs), total=len(seqs)))

    return np.array(embs).squeeze(), seqs


# def compute_all_emb(model, X):
#     transformer_model = transformers.AutoModel.from_pretrained(model.checkpoint).to(
#         DEVICE
#     )
#     tokenizer_embeddings = transformers.AutoTokenizer.from_pretrained(model.checkpoint)
#     process_item_with_args = partial(
#         compute_emb,
#         model=model,
#         transformer_model=transformer_model,
#         tokenizer_embeddings=tokenizer_embeddings,
#     )
#     with Pool(4) as pool:
#         res = list(tqdm(pool.imap(process_item_with_args, X), total=len(X)))

#     embs = []
#     for r in res:
#         for emb in r:
#             embs.append(emb[0])

#     return embs


def predict_from_seqs(model, seqs):
    pred = 0
    for seq in seqs:
        if seq in model.coefs_dict_:
            pred += model.coefs_dict_[seq]
        else:
            n_unseen_ngrams += 1

    if isinstance(model, RegressorMixin):
        return pred
    elif isinstance(model, ClassifierMixin):
        return ((pred + model.linear.intercept_) > 0).astype(int)
    return


def sparse_predict(model, X, warn, sparsity=0):
    """Predict only the cached coefs in self.coefs_dict_"""
    assert hasattr(model, "coefs_dict_"), "coefs are not cached!"
    preds = []
    lengths = []
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

        pred_list = sorted(pred_list, key=abs, reverse=True)

        if len(pred_list) == 0:
            preds.append(0)
        elif sparsity > 0:
            # only look at the top sparsity fraction of seqs
            idx = np.where(
                np.cumsum(np.abs(pred_list)) / sum(np.abs(pred_list)) > sparsity
            )[0][0]
            preds.append(sum(pred_list[:idx]))
            lengths.append((idx + 1) / len(pred_list))
        else:
            preds.append(sum(pred_list))
            lengths.append(1)

    if n_unseen_ngrams > 0 and warn:
        warnings.warn(
            f"Saw an unseen ungram {n_unseen_ngrams} times. \
For better performance, call cache_linear_coefs on the test dataset \
before calling predict."
        )

    if isinstance(model, RegressorMixin):
        return preds, lengths
    elif isinstance(model, ClassifierMixin):
        return ((preds + model.linear.intercept_) > 0).astype(int), lengths


if __name__ == "__main__":
    for dataset in dsets.keys():
        checkpoint = dsets[dataset]

        # set up data
        dset, dataset_key_text = data.process_data_and_args(dataset)
        dset_train = dset["train"]
        dset_val = dset["validation"]

        # load model
        with open(f"../results/4gram_{dataset}_imodelsx.pkl", "rb") as f:
            m = pkl.load(f)

        # predict
        m.cache_linear_coefs(dset_val[dataset_key_text])

        transformer_model = transformers.AutoModel.from_pretrained(m.checkpoint).to(
            DEVICE
        )
        tokenizer_embeddings = transformers.AutoTokenizer.from_pretrained(m.checkpoint)

        # compute all embeddings
        preds = []
        for x in dset_val[dataset_key_text][:100]:
            embs, seqs = compute_emb(
                x,
                model=m,
                transformer_model=transformer_model,
                tokenizer_embeddings=tokenizer_embeddings,
            )

            breakpoint()
            u, s, vh = np.linalg.svd(embs)
            # get 80% rank approximation
            n_comps = np.where(np.cumsum(s) / np.sum(s) < 0.8)[0][-1]
            approx = u[:, :n_comps] @ np.diag(s[:n_comps]) @ vh[:n_comps, :]
            # get the indices of the n_comps smallest elements of np.abs((embs - approx) @ w)
            indices = [
                idx
                for idx, val in sorted(
                    enumerate(np.abs((embs - approx) @ m.linear.coef_.reshape(-1))),
                    key=lambda x: x[1],
                )[:n_comps]
            ]

            preds.append(predict_from_seqs(m, [seqs[i] for i in indices])[0])

            # svd = TruncatedSVD(n_components=768)
            # svd.fit(np.array(embs))
            # top_k_singular_values = svd.singular_values_

        print("Accuracy: ", np.mean(preds == dset_val["label"][:100]))
        breakpoint()

        # generate plot
        sparsity_levels = [0.001, 0.01, 0.1, 0.2, 0.5, 0.8, 0.95]
        acc_vals = []
        avg_frac_ngrams = []
        for sparsity in sparsity_levels:
            preds, lengths = sparse_predict(
                m, dset_val[dataset_key_text], True, sparsity=sparsity
            )
            acc_vals.append(np.mean(preds == dset_val["label"]))
            avg_frac_ngrams.append(np.mean(lengths))
            print("sparsity level", sparsity, "acc_val", acc_vals[-1])
            print("avg frac of ngrams", avg_frac_ngrams[-1])

        plt.figure()
        plt.plot(sparsity_levels, acc_vals)
        plt.xlabel("Sparsity Level")
        plt.ylabel("Validation Acc")
        plt.savefig(f"{dataset}_sparsity.pdf", dpi=200)

        plt.figure()
        plt.plot(avg_frac_ngrams, acc_vals)
        plt.xlabel("fraction ngrams")
        plt.ylabel("Validation Acc")
        plt.savefig(f"{dataset}_ngrams.pdf", dpi=200)
