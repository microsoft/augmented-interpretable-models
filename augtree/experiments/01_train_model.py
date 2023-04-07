import argparse
import inspect
import logging
import os
import pickle as pkl
import random
from collections import defaultdict
from copy import deepcopy
from os.path import dirname, join

import cache_save_utils
import datasets
import imodels
import imodelsx
import imodelsx.augtree as llm_tree
import numpy as np
import torch
from imodelsx.metrics import (
    metrics_classification_discrete,
    metrics_classification_proba,
    metrics_regression,
)
from sklearn.ensemble import BaggingClassifier, BaggingRegressor
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from mimic.utils import get_mimic_X_y, load_mimic_data, split_mimic_data

datasets.logging.set_verbosity_error()
path_to_repo = dirname(dirname(os.path.abspath(__file__)))


def get_model(args):
    if args.model_name == "llm_tree":
        LLM_PROMPT_CONTEXTS = {
            "sst2": " in the context of movie reviews",
            "rotten_tomatoes": " in the context of movie reviews",
            "imdb": " in the context of movie reviews",
            "financial_phrasebank": " in the context of financial sentiment",
            "ag_news": " in the context of news headlines",
            "tweet_eval": " in the context of tweets",
            "emotion": " in the context of tweet sentiment",
        }
        if args.use_llm_prompt_context:
            llm_prompt_context = LLM_PROMPT_CONTEXTS[args.dataset_name]
        else:
            llm_prompt_context = ""
        if args.refinement_strategy == "embs":
            embs_manager = imodelsx.augtree.embed.EmbsManager(
                dataset_name=args.dataset_name,
                ngrams=args.ngrams,
                # metric=args.embs_refine_metric,
            )
        else:
            embs_manager = None
        if args.classification_or_regression == "classification":
            cls = imodelsx.augtree.augtree.AugTreeClassifier
        else:
            cls = imodelsx.augtree.augtree.AugTreeRegressor
        model = cls(
            max_depth=args.max_depth,
            max_features=args.max_features,
            refinement_strategy=args.refinement_strategy,
            split_strategy=args.split_strategy,
            use_refine_ties=args.use_refine_ties,
            llm_prompt_context=llm_prompt_context,
            verbose=args.use_verbose,
            embs_manager=embs_manager,
            use_stemming=args.use_stemming,
            cache_expansions_dir=args.cache_expansions_dir,
        )
    elif args.model_name == "decision_tree":
        if args.classification_or_regression == "classification":
            model = DecisionTreeClassifier(max_depth=args.max_depth)
        else:
            model = DecisionTreeRegressor(max_depth=args.max_depth)
    elif args.model_name == "c45":
        model = imodels.C45TreeClassifier(max_rules=int(2**args.max_depth) - 1)
    elif args.model_name == "id3":
        model = DecisionTreeClassifier(max_depth=args.max_depth, criterion="entropy")
    elif args.model_name == "hstree":
        estimator_ = DecisionTreeClassifier(max_depth=args.max_depth)
        model = imodels.HSTreeClassifier(estimator_=estimator_)
    elif args.model_name == "ridge":
        model = RidgeClassifier(alpha=args.alpha)
    elif args.model_name == "rule_fit":
        model = imodels.RuleFitClassifier(max_rules=args.max_rules)
    elif args.model_name == "linear_finetune":
        if args.classification_or_regression == "classification":
            model = imodelsx.LinearFinetuneClassifier()
        else:
            model = imodelsx.LinearFinetuneRegressor()
    else:
        raise ValueError(f"Invalid model_name: {args.model_name}")

    # make baggingclassifier
    if args.n_estimators > 1:
        if args.model_name == "llm_tree":
            model = imodelsx.augtree.ensemble.BaggingEstimatorText(
                model, n_estimators=args.n_estimators
            )
        elif args.model_name == "decision_tree":
            if args.classification_or_regression == "classification":
                model = BaggingClassifier(model, n_estimators=args.n_estimators)
            else:
                model = BaggingRegressor(model, n_estimators=args.n_estimators)

    return model


# initialize args
def add_main_args(parser):
    """Caching uses the non-default values from argparse to name the saving directory.
    Changing the default arg an argument will break cache compatibility with previous runs.
    """

    # dataset args
    parser.add_argument(
        "--dataset_name",
        type=str,  # csinva/fmri_language_responses
        default="rotten_tomatoes",
        help="name of dataset",
    )
    parser.add_argument(
        "--subsample_frac", type=float, default=1, help="fraction of samples to use"
    )
    parser.add_argument(
        "--ngrams", type=int, default=2, help="ngram range for tokenization"
    )
    parser.add_argument(
        "--use_stemming",
        type=int,
        default=0,
        choices=[0, 1],
        help="whether to use stemming",
    )
    parser.add_argument(
        "--label_name",
        type=str,
        default="label",
        help="name of the label (default label), for fmri, might be voxel_0",
    )

    # training misc args
    parser.add_argument("--seed", type=int, default=1, help="random seed")
    parser.add_argument(
        "--save_dir",
        type=str,
        default=join(path_to_repo, "results", "tmp"),
        help="directory for saving",
    )

    # model args
    parser.add_argument(
        "--model_name",
        type=str,
        choices=[
            "llm_tree",
            "decision_tree",
            "id3",
            "hstree",
            "c45",
            "ridge",
            "rule_fit",
        ],
        default="llm_tree",
        help="name of model",
    )
    parser.add_argument(
        "--n_estimators",
        type=int,
        default=1,
        help="number of estimators when ensembling (will ensemble with bagging)",
    )
    parser.add_argument(
        "--max_depth", type=int, default=3, help="max depth of tree (llm_tree)"
    )
    parser.add_argument(
        "--split_strategy", type=str, default="cart", help="split strategy (llm_tree)"
    )
    parser.add_argument(
        "--max_features",
        type=int,
        default=1,
        help="max features to include in a single split (llm_tree)",
    )
    parser.add_argument(
        "--refinement_strategy",
        type=str,
        default="llm",
        choices=["None", "llm", "embs"],
        help="strategy to use to refine keywords (llm_tree)",
    )
    parser.add_argument(
        "--use_refine_ties",
        type=int,
        default=0,
        choices=[0, 1],
        help="whether to keep ties llm refine keywords (llm_tree). This ends up being pretty minor",
    )
    parser.add_argument(
        "--use_llm_prompt_context",
        type=int,
        default=0,
        choices=[0, 1],
        help="whether to use llm prompt context (llm_tree)",
    )
    # parser.add_argument('--embs_refine_metric', type=str, default='euclidean', choices=['euclidean', 'cosine'],
    # help='which metric to use for embedding refinement (llm_tree)')
    # parser.add_argument('--embs_checkpoint', type=str, default='bert-base-uncased',
    # help='checkpoint to use for embedding refinement (llm_tree)')

    # baseline model args
    parser.add_argument(
        "--max_leaf_nodes",
        type=int,
        default=2,
        help="max number of leaf nodes (decision tree)",
    )
    parser.add_argument(
        "--alpha", type=float, default=1, help="regularization strength (ridge)"
    )
    parser.add_argument(
        "--max_rules", type=int, default=2, help="max number of rules (rule fit)"
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
        "--use_verbose",
        type=int,
        default=1,
        choices=[0, 1],
        help="whether to print verbosely",
    )
    parser.add_argument(
        '--cache_expansions_dir',
        type=str,
        default = join(path_to_repo, 'results', 'gpt3_cache'),
    )
    return parser


if __name__ == "__main__":
    # get args
    parser = argparse.ArgumentParser()
    parser_without_computational_args = add_main_args(parser)
    parser = add_computational_args(deepcopy(parser_without_computational_args))
    args = parser.parse_args()

    # set up logging
    logger = logging.getLogger()
    if args.use_verbose:
        logging.basicConfig(level=logging.INFO)
    else:
        logging.basicConfig(level=logging.ERROR)

    # set up saving directory + check for cache
    already_cached, save_dir_unique = cache_save_utils.get_save_dir_unique(
        parser, parser_without_computational_args, args, args.save_dir
    )
    if args.use_cache and already_cached:
        logging.info(f"cached version exists! Successfully skipping :)\n\n\n")
        exit(0)
    if args.use_verbose:
        for k in sorted(vars(args)):
            logger.info("\t" + k + " " + str(vars(args)[k]))
        logging.info(f"\n\n\tsaving to " + save_dir_unique + "\n")

    # set seed
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    tokenizer = llm_tree.utils.get_spacy_tokenizer(use_stemming=args.use_stemming)

    # load text data
    if args.dataset_name == "mimic":
        mimic_path = "/root/.data/llmtree/"
        mimic_data, vocab = load_mimic_data(path=mimic_path)
        X, y = get_mimic_X_y(mimic_data, label="mortality")
        counts_filename = os.path.join(mimic_path, "mimiciv-2.2-counts.pkl")
        if not os.path.exists(counts_filename):
            (
                X_counts_array,
                _,
                feature_names,
            ) = llm_tree.data.convert_text_data_to_counts_array(
                X_train=X,
                X_test=[],
                ngrams=args.ngrams,
                tokenizer=tokenizer,
                vocabulary=vocab,
            )
            counts_dict = {"counts": X_counts_array, "feature_names": feature_names}
            with open(counts_filename, "wb") as f:
                pkl.dump(counts_dict, f)
        else:
            with open(counts_filename, "rb") as f:
                counts_dict = pkl.load(f)

        feature_names = counts_dict["feature_names"]
        X_train, X_train_text, X_test, X_test_text, y_train, y_test = split_mimic_data(
            X_counts=counts_dict["counts"],
            X=X,
            y=y,
            test_size=0.33,
            subsample_frac=args.subsample_frac,
        )
        print("done splitting data")
    else:
        (
            X_train_text,
            X_test_text,
            y_train,
            y_test,
        ) = imodelsx.data.load_huggingface_dataset(
            dataset_name=args.dataset_name,
            subsample_frac=args.subsample_frac,
            return_lists=True,
            binary_classification=True,
            label_name=args.label_name,
        )
        (
            X_train,
            X_test,
            feature_names,
        ) = llm_tree.data.convert_text_data_to_counts_array(
            X_train_text,
            X_test_text,
            ngrams=args.ngrams,
            tokenizer=tokenizer,
        )

    X_train, X_cv, X_train_text, X_cv_text, y_train, y_cv = train_test_split(
        X_train, X_train_text, y_train, test_size=0.33, random_state=args.seed
    )
    if args.dataset_name == "csinva/fmri_language_responses":
        args.classification_or_regression = "regression"
        X_train_text = [" ".join(x.split()[-24:-4]) for x in X_train_text]
        X_test_text = [" ".join(x.split()[-24:-4]) for x in X_test_text]
    else:
        args.classification_or_regression = "classification"

    # load model
    model = get_model(args)

    # set up saving dictionary + save params file
    r = defaultdict(list)
    r.update(vars(args))
    r["save_dir_unique"] = save_dir_unique
    r["feature_names"] = feature_names
    cache_save_utils.save_json(
        args=args, save_dir=save_dir_unique, fname="params.json", r=r
    )
    # print('data type', args.classification_or_regression)
    # print('shapes', X_train.shape, X_cv.shape, X_test.shape)
    # print('y', np.unique(y_train), np.unique(y_cv), np.unique(y_test))

    # fit the model
    print("fitting model...")
    fit_parameters = inspect.signature(model.fit).parameters.keys()
    kwargs = {}
    if (
        "feature_names" in fit_parameters
        and feature_names is not None
        and not isinstance(model, imodels.C45TreeClassifier)
    ):
        kwargs["feature_names"] = feature_names
    if "X_text" in fit_parameters:
        kwargs["X_text"] = X_train_text
    if "X" in fit_parameters:
        kwargs["X"] = X_train
    kwargs["y"] = y_train
    model.fit(**kwargs)

    # evaluate
    if args.use_verbose:
        logging.info("evaluating...")
    for split_name, (X_text_, X_, y_) in zip(
        ["train", "cv", "test"],
        [
            (X_train_text, X_train, y_train),
            (X_cv_text, X_cv, y_cv),
            (X_test_text, X_test, y_test),
        ],
    ):
        predict_parameters = inspect.signature(model.predict).parameters.keys()
        if "X_text" in predict_parameters:
            y_pred_ = model.predict(X_text=X_text_)
        else:
            y_pred_ = model.predict(X_)

        if args.classification_or_regression == "classification":
            y_pred_ = y_pred_.astype(int)

            # classification metrics discrete
            for metric_name, metric_fn in metrics_classification_discrete.items():
                r[f"{metric_name}_{split_name}"] = metric_fn(y_, y_pred_)

            # classification metrics proba
            if hasattr(model, "predict_proba"):
                if "X_text" in predict_parameters:
                    y_pred_proba_ = model.predict_proba(X_text=X_text_)[:, 1]
                else:
                    y_pred_proba_ = model.predict_proba(X_)[:, 1]
                for metric_name, metric_fn in metrics_classification_proba.items():
                    r[f"{metric_name}_{split_name}"] = metric_fn(y_, y_pred_proba_)

        elif args.classification_or_regression == "regression":
            # regression metrics
            for metric_name, metric_fn in metrics_regression.items():
                r[f"{metric_name}_{split_name}"] = metric_fn(y_, y_pred_)

    # save results
    pkl.dump(r, open(join(save_dir_unique, "results.pkl"), "wb"))
    if args.model_name == "linear_finetune":
        model.model = None
        model.tokenizer = None
    pkl.dump(model, open(join(save_dir_unique, "model.pkl"), "wb"))
    logging.info("Succesfully completed :)\n\n")
