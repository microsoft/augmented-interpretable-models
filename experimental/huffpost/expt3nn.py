import os
import pickle as pkl

import numpy as np
import torch
from sklearn.pipeline import Pipeline
from skorch import NeuralNetClassifier
from skorch.callbacks import LRScheduler, ProgressBar
from skorch.hf import HuggingfacePretrainedTokenizer
from torch import nn
from torch.optim.lr_scheduler import LambdaLR
import transformers
from tqdm import tqdm

from preprocess import clean_headlines, sample_data

# Choose a tokenizer and BERT model that work together
TOKENIZER = "distilbert-base-uncased"
PRETRAINED_MODEL = "distilbert-base-uncased"

# model hyper-parameters
OPTMIZER = torch.optim.AdamW
LR = 5e-5
MAX_EPOCHS = 10
CRITERION = nn.CrossEntropyLoss
BATCH_SIZE = 32

# device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def embed(X, tokenizer, model):
    tokens = tokenizer(X, padding=True, truncation=True, return_tensors="pt")
    tokens = tokens.to(model.device)
    output = model(**tokens)
    embs = output["last_hidden_state"].cpu().detach().numpy().mean(axis=1)

    return embs


class ArticleNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Linear(768, 11)

    def forward(self, x):
        return self.model(x)


def create_pipeline(lr_schedule):
    pipeline = Pipeline(
        [
            (
                "net",
                NeuralNetClassifier(
                    ArticleNetwork,
                    optimizer=OPTMIZER,
                    lr=LR,
                    max_epochs=MAX_EPOCHS,
                    criterion=CRITERION,
                    batch_size=BATCH_SIZE,
                    iterator_train__shuffle=True,
                    device=DEVICE,
                    callbacks=[
                        LRScheduler(
                            LambdaLR, lr_lambda=lr_schedule, step_every="batch"
                        ),
                        ProgressBar(),
                    ],
                ),
            ),
        ]
    )

    return pipeline


if __name__ == "__main__":
    # load data
    with open(f"Data/huffpost.pkl", "rb") as f:
        huffpost_data = pkl.load(f)

    tokenizer = transformers.AutoTokenizer.from_pretrained(TOKENIZER)
    model = transformers.AutoModel.from_pretrained(PRETRAINED_MODEL).to("cpu")
    huffpost_data = clean_headlines(huffpost_data)
    for year in [2012, 2013, 2014, 2015, 2016, 2017, 2018]:
        print("Year ", year)
        d = {"X_id": [], "y_id": [], "X_od": [], "y_od": []}
        for in_dist in [True, False]:
            data, labels = sample_data(huffpost_data, year, in_dist=in_dist, frac=1)
            X = []
            for x in tqdm(data):
                X.append(embed(x, tokenizer, model))
            if in_dist:
                key_data, key_labels = "X_id", "y_id"
            else:
                key_data, key_labels = "X_od", "y_od"

            d[key_data] = np.array(X)
            d[key_labels] = np.array(labels)
            
        # save embeddings
        pkl.dump(d, open(f"Data/emb_{year}.pkl", "wb"))

    num_training_steps = MAX_EPOCHS * (len(data) // BATCH_SIZE + 1)

    def lr_schedule(current_step):
        factor = float(num_training_steps - current_step) / float(
            max(1, num_training_steps)
        )
        assert factor > 0
        return factor

        breakpoint()
    pipeline = create_pipeline(lr_schedule=lr_schedule)
    pipeline.fit(X, y)

    # save model
    if not os.path.exists("models/expt3"):
        os.makedirs("models/expt3")
    pkl.dump(pipeline, open(f"models/expt3/expt3nn.pkl", "wb"))
