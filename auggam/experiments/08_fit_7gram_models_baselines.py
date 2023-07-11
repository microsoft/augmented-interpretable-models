from imodelsx import AugGAMClassifier
import numpy as np
import fire
from auggam import data
from os.path import join, dirname, abspath
import imodelsx
import joblib

path_to_repo = dirname(dirname(abspath(__file__)))


def run_dataset(
    checkpoint="hkunlp/instructor-xl",
    dataset: str = "financial_phrasebank",
    ngrams=7,
):
    # set up data
    dset, dataset_key_text = data.process_data_and_args(dataset)
    dset_train = dset["train"]
    dset_val = dset["validation"]
    # dset_train = dset_train.filter(lambda example, indice: indice <100, with_indices=True)
    # dset_val = dset_val.filter(lambda example, indice: indice <100, with_indices=True)

    if "vectorizer" in checkpoint:
        m = imodelsx.LinearNgramClassifier(
            checkpoint=checkpoint, all_ngrams=True, ngrams=ngrams, random_state=42
        )
    elif checkpoint == "linear_finetune":
        m = imodelsx.LinearFinetuneClassifier(
            checkpoint="bert-base-uncased",
            all_ngrams=True,
            ngrams=ngrams,
            random_state=42,
        )
    else:
        if checkpoint == "hkunlp/instructor-xl":
            INSTRUCTIONS = {
                "rotten_tomatoes": "Represent the Review sentence for classifying emotion as positive or negative; Input:",
                "sst2": "Represent the Review sentence for classifying emotion as positive or negative; Input:",
                "emotion": "Represent the Tweet for classifying emotion as positive or negative; Input:",
                "financial_phrasebank": "Represent the Financial statement for classifying emotion as positive or negative; Input:",
            }
            instructor_prompt = INSTRUCTIONS[dataset]
        else:
            instructor_prompt = None

        # fit model
        m = AugGAMClassifier(
            checkpoint=checkpoint,
            all_ngrams=True,
            ngrams=ngrams,
            random_state=42,
            instructor_prompt=instructor_prompt,
        )

    print("\n\t++++++++Fitting++++++++\n")
    m.fit(dset_train[dataset_key_text], dset_train["label"], verbose=True)

    # predict
    # preds = m.predict(dset_val[dataset_key_text])
    # print('acc_val', np.mean(preds == dset_val['label']))
    print("\n\t++++++++Caching++++++++\n")
    m.cache_linear_coefs(dset_val[dataset_key_text])

    print("\n\t++++++++Predicting++++++++\n")
    preds = m.predict(dset_val[dataset_key_text])
    acc_val = np.mean(preds == dset_val["label"])
    print(dataset, "acc_val", acc_val)

    out_dir = join(path_to_repo, "results", "7gram")
    os.makedirs(out_dir, exist_ok=True)
    joblib.dump(
        m,
        join(out_dir, f"{checkpoint.replace('/', '_')}_{dataset}_imodelsx.pkl"),
    )
    joblib.dump(
        {"acc_val": acc_val},
        join(out_dir, f"{checkpoint.replace('/', '_')}_acc_{dataset}_imodelsx.pkl"),
    )


if __name__ == "__main__":
    fire.Fire(run_dataset)
