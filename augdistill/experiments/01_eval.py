import datasets
import argparse
from copy import deepcopy
import logging
import random
from collections import defaultdict
from os.path import join
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
import joblib
import imodels
import inspect
import torch
import os.path
import imodelsx.cache_save_utils
from imodelsx import AugLinearClassifier

path_to_repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

EMBEDDING_STRING_SETTINGS = {
    'instructor_sentiment': ('Represent the movie review for sentiment classification: ', ''),
    'synonym': ('A synonym of the phrase ', ' is'),
    'movie_sentiment': ('In a movie review, the sentiment of the phrase "', '" is'),
}


# initialize args
def add_main_args(parser):
    """Caching uses the non-default values from argparse to name the saving directory.
    Changing the default arg an argument will break cache compatibility with previous runs.
    """

    # dataset args
    parser.add_argument(
        "--dataset_name", type=str, default="rotten_tomatoes", help="name of dataset"
    )
    parser.add_argument(
        '--checkpoint', type=str, default='gpt2', help='checkpoint for the model'
    )
    parser.add_argument(
        "--ngrams", type=int, default=2, help="ngrams"
    )
    parser.add_argument(
        "--use_all_ngrams", type=int, default=1, choices=[0, 1], help="whether to use all ngrams"
    )
    parser.add_argument(
        "--embedding_ngram_strategy", type=str, default='mean', choices=['mean', 'next_token_distr'], help="strategy to compute ngram embeddings"
    )
    parser.add_argument(
        "--embedding_string_prompt", type=str, default="synonym", choices=set(list(EMBEDDING_STRING_SETTINGS.keys()) + ['None']), help="key for embedding string"
    )
    parser.add_argument(
        '--zeroshot_strategy', type=str, default='pos_class', choices=['pos_class', 'difference'], help='strategy for zeroshot'
    )
    parser.add_argument(
        '--renormalize_embs_strategy', type=str, default='StandardScaler',
        choices=[None, 'None', 'StandardScaler', 'QuantileTransformer'], help='strategy for renormalizing embeddings'
    )
    # training misc args
    parser.add_argument("--seed", type=int, default=1, help="random seed")
    parser.add_argument(
        "--save_dir",
        type=str,
        default=join(path_to_repo, "results"),
        help="directory for saving",
    )

    return parser


def add_computational_args(parser):
    """Arguments that only affect computation and not the results (shouldnt use when checking cache)"""
    parser.add_argument(
        "--use_cache",
        type=int,
        default=1,
        choices=[0, 1],
        help="whether to check for cache",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="batch_size for computing embeddings",
    )
    return parser


if __name__ == "__main__":
    # get args
    parser = argparse.ArgumentParser()
    parser_without_computational_args = add_main_args(parser)
    parser = add_computational_args(
        deepcopy(parser_without_computational_args))
    args = parser.parse_args()

    # set up logging
    logger = logging.getLogger()
    logging.basicConfig(level=logging.INFO)

    # set up saving directory + check for cache
    already_cached, save_dir_unique = imodelsx.cache_save_utils.get_save_dir_unique(
        parser, parser_without_computational_args, args, args.save_dir
    )

    if args.use_cache and already_cached:
        logging.info(f"cached version exists! Successfully skipping :)\n\n\n")
        exit(0)
    for k in sorted(vars(args)):
        logger.info("\t" + k + " " + str(vars(args)[k]))
    logging.info(f"\n\n\tsaving to " + save_dir_unique + "\n")

    # set seed
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # load text data
    dset_val = datasets.load_dataset(args.dataset_name)['validation']
    dset_val = dset_val.select(np.random.choice(
        len(dset_val), size=150, replace=False))

    # load model
    embedding_prefix, embedding_suffix = EMBEDDING_STRING_SETTINGS.get(
        args.embedding_string_prompt, ("", ""))
    m = AugLinearClassifier(
        checkpoint=args.checkpoint,
        embedding_ngram_strategy=args.embedding_ngram_strategy,
        embedding_prefix=embedding_prefix,
        embedding_suffix=embedding_suffix,
        ngrams=args.ngrams,
        all_ngrams=args.use_all_ngrams,  # also use lower-order ngrams
        zeroshot_class_dict={0: ['negative', 'boring', 'awful'],
                             1: ['positive', 'great', 'good']},
        prune_stopwords=True,
    )

    # set up saving dictionary + save params file
    r = defaultdict(list)
    r.update(vars(args))
    r["save_dir_unique"] = save_dir_unique
    imodelsx.cache_save_utils.save_json(
        args=args, save_dir=save_dir_unique, fname="params.json", r=r
    )

    # fit
    m.fit(None, [0, 1], verbose=True, batch_size=args.batch_size)
    m.cache_linear_coefs(
        dset_val['text'], renormalize_embs_strategy=args.renormalize_embs_strategy, verbose=True, batch_size=args.batch_size)

    # evaluate
    preds = m.predict(dset_val['text'])
    # m._predict_cached(dset_val['text'], warn=False)
    preds_proba = m.predict_proba(dset_val['text'])
    r['roc_val'] = roc_auc_score(
        dset_val['label'], preds_proba[:, 1]).round(2)
    r['acc_val'] = np.mean(preds == dset_val['label'])
    r['acc_baseline'] = np.mean(dset_val['label'])
    r['mean_pred'] = np.mean(preds)
    r['mean_pred_proba'] = np.mean(preds_proba[:, 1])

    # save results
    joblib.dump(
        r, join(save_dir_unique, "results.pkl")
    )  # caching requires that this is called results.pkl
    # joblib.dump(model, join(save_dir_unique, "model.pkl"))
    # print(r)
    logging.info("Succesfully completed :)\n\n")
