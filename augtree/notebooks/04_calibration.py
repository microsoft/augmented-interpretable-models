import matplotlib.pyplot as plt
from os.path import join
import pickle as pkl
import imodelsx.process_results
import sys
import dvu
import numpy as np
import viz
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.calibration import calibration_curve
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
import imodelsx
from sklearn.metrics import roc_auc_score, accuracy_score
results_dir = '../results/feb11/'

r = imodelsx.process_results.get_results_df(results_dir, use_cached=True)
out = defaultdict(list)
for dataset_name in tqdm(['financial_phrasebank', 'emotion', 'sst2', 'rotten_tomatoes']):
    rd = r[r.dataset_name == dataset_name]

    # get data
    X_train_text, X_test_text, y_train, y_test = imodelsx.data.load_huggingface_dataset(
        dataset_name=dataset_name,
        return_lists=True,
        binary_classification=True,
    )
    X_train_text, X_cv_text, y_train, y_cv = train_test_split(
        X_train_text, y_train, test_size=0.33, random_state=1)

    # load tree
    preds_proba = np.zeros((y_test.size, 2))
    preds_proba_cv = np.zeros((y_cv.size, 2))

    for seed in [1, 2, 3]:
        rdt = rd[rd.n_estimators == 40]
        rdt = rdt[rdt.model_name == 'llm_tree']
        n_rdt = rdt.shape[0]
        if seed in rdt.seed.values:
            rdt = rdt[rdt.seed == seed]
            rdt
            assert rdt.shape[0] == 1
            rdt = rdt.iloc[0]
            model = pkl.load(open(join(rdt.save_dir_unique, 'model.pkl'), 'rb'))
            preds_proba += model.predict_proba(X_text=X_test_text)
            preds_proba_cv += model.predict_proba(X_text=X_cv_text)
    preds_proba /= n_rdt
    preds_proba_cv /= n_rdt
    preds = np.argmax(preds_proba, axis=1)

    # get linear-finetune preds
    rdn = rd[rd.model_name == 'linear_finetune']
    assert rdn.shape[0] == 1
    rdn = rdn.iloc[0]
    model = pkl.load(open(join(rdn.save_dir_unique, 'model.pkl'), 'rb'))
    model._initialize_checkpoint_and_tokenizer()
    preds_proba_dnn = model.predict_proba(X_text=X_test_text)
    preds_dnn = np.argmax(preds_proba_dnn, axis=1)

    # get acc
    acc = (preds == y_test).mean()
    acc_dnn = np.mean(np.array(preds_dnn) == y_test)
    print(f'acc {acc:0.2f} acc_dnn {acc_dnn:0.2f}')

    # get joint performance
    args = np.argsort(np.abs(preds_proba.max(axis=1)))[::-1]
    preds_proba_sort = preds_proba[args].tolist()
    preds_proba_sort_dnn = preds_proba_dnn[args].tolist()
    preds_proba_combined = [np.array(preds_proba_sort[:i] + preds_proba_sort_dnn[i:]).reshape(-1, 2) for i in range(len(preds_proba_sort))]
    rocs = [roc_auc_score(y_test[args], p[:, 1]) for p in preds_proba_combined]
    accs = [accuracy_score(y_test[args], np.argmax(p, axis=1)) for p in preds_proba_combined]
    

    out['dset'].append(dataset_name)
    out['acc'].append(acc)
    out['acc_dnn'].append(acc_dnn)
    out['rocs'].append(rocs)
    out['accs'].append(accs)
    out['preds_proba'].append(preds_proba)
    out['preds_proba_cv'].append(preds_proba_cv)
    out['y_test'].append(y_test)
    out['y_cv'].append(y_cv)
    pkl.dump(out, open('../results/acc_interpolate.pkl', 'wb'))