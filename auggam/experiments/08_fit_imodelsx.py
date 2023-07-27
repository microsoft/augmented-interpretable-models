from imodelsx import AugGAMClassifier
import numpy as np
import fire
from auggam import data
from os.path import join, dirname, abspath
import imodelsx
import joblib
import os
from nltk.tokenize import word_tokenize
path_to_repo = dirname(dirname(abspath(__file__)))


def run_dataset(
    checkpoint="hkunlp/instructor-xl",
    dataset: str = "financial_phrasebank",
    ngrams=7,
    subsample_n=10_000,
    use_simple_tokenizer=0,
):
    out_dir = join(path_to_repo, "results", "7gram")
    ngram_str = "" if ngrams == 7 else f"ngram{ngrams}_"
    tokenizer_str = "" if not use_simple_tokenizer else "tokenizer=simplified"
    out_file = join(
        out_dir, f"{checkpoint.replace('/', '_')}_acc_{dataset}_{ngram_str}{tokenizer_str}imodelsx.pkl"
    )
    if os.path.exists(out_file):
        print(f"\nCached {out_file} :)\n")
        return
    if use_simple_tokenizer:
        tokenizer_ngrams = word_tokenize
        # tokenizer_ngrams = lambda x: x.split()
    else:
        tokenizer_ngrams = None

    # set up data
    dset, dataset_key_text = imodelsx.data.load_huggingface_dataset(dataset)

    dset_train = dset["train"]
    dset_val = dset["validation"]

    if not dataset in ["financial_phrasebank", "sst2", "emotion", "rotten_tomatoes"]:
        if subsample_n < len(dset_train):
            idxs = np.random.choice(range(len(dset_train)), size=subsample_n)
            dset_train = dset_train.select(idxs)

    kwargs = {}
    if "vectorizer" in checkpoint:
        m = imodelsx.LinearNgramClassifier(
            checkpoint=checkpoint, all_ngrams=True, ngrams=ngrams, random_state=42
        )
    elif checkpoint == "linear_finetune":
        m = imodelsx.LinearFinetuneClassifier(
            checkpoint="bert-base-uncased",
            random_state=42,
        )
    else:
        if checkpoint == "hkunlp/instructor-xl":
            INSTRUCTIONS = {
                "rotten_tomatoes": "Represent the Review sentence for classifying emotion as positive or negative; Input:",
                "sst2": "Represent the Review sentence for classifying emotion as positive or negative; Input:",
                "emotion": "Represent the Tweet for classifying emotion as positive or negative; Input:",
                "financial_phrasebank": "Represent the Financial statement for classifying emotion as positive or negative; Input:",
                "ag_news": "Represent the news article for classifying its category; Input:",
                "dbpedia_14": "Represent the sentence for classifying ontology; Input:",
                "trec": "Represent the question for classifying its category; Input:",
            }
            instructor_prompt = INSTRUCTIONS.get(dataset, "Represent the question for classifying its category; Input:")
        else:
            instructor_prompt = None

        # fit model
        m = AugGAMClassifier(
            checkpoint=checkpoint,
            all_ngrams=True,
            ngrams=ngrams,
            random_state=42,
            instructor_prompt=instructor_prompt,
            tokenizer_ngrams=tokenizer_ngrams,
        )
        kwargs["batch_size"] = 32

    print("\n\t++++++++Fitting++++++++\n")
    m.fit(dset_train[dataset_key_text], dset_train["label"], verbose=True, **kwargs)

    # predict
    # preds = m.predict(dset_val[dataset_key_text])
    # print('acc_val', np.mean(preds == dset_val['label']))
    if isinstance(m, AugGAMClassifier):
        print("\n\t++++++++Caching++++++++\n")
        m.cache_linear_coefs(dset_val[dataset_key_text])

    print("\n\t++++++++Predicting++++++++\n")
    preds = m.predict(dset_val[dataset_key_text])
    acc_val = np.mean(preds == dset_val["label"])
    print(dataset, "acc_val", acc_val)

    os.makedirs(out_dir, exist_ok=True)
    joblib.dump(
        m,
        join(out_dir, f"{checkpoint.replace('/', '_')}_{dataset}_imodelsx.pkl"),
    )
    joblib.dump({"acc_val": acc_val, "ngrams": ngrams}, out_file)


if __name__ == "__main__":
    fire.Fire(run_dataset)
