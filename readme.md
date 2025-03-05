<h1 align="center">   <img src="https://microsoft.github.io/augmented-interpretable-models/auggam_gif.gif" width="25%"><img src="https://microsoft.github.io/augmented-interpretable-models/logo.svg?sanitize=True&kill_cache=1" width="45%"> <img src="https://microsoft.github.io/augmented-interpretable-models/auggam_gif.gif" width="25%"></h1>
<p align="center"> Augmenting Interpretable Models with LLMs during Training
</p>

<p align="center">
  <img src="https://img.shields.io/badge/license-mit-blue.svg">
  <img src="https://img.shields.io/badge/python-3.6+-blue">
  <img src="https://img.shields.io/badge/pytorch-1.0+-blue">
  <img src="https://img.shields.io/pypi/v/imodelsx?color=green">  
</p>  

<p align="center">
  <img src="https://microsoft.github.io/augmented-interpretable-models/ovw.png" width="60%">
</p>  

This repo contains code to reproduce the experiments in the Aug-imodels paper ([Nature Communications, 2023](https://arxiv.org/abs/2209.11799)). For a simple scikit-learn interface to use Aug-imodels, use the [imodelsX library](https://github.com/csinva/imodelsX). Below is a quickstart example.

Installation: `pip install imodelsx`

```python
from imodelsx import AugLinearClassifier, AugTreeClassifier, AugLinearRegressor, AugTreeRegressor
import datasets
import numpy as np

# set up data
dset = datasets.load_dataset('rotten_tomatoes')['train']
dset = dset.select(np.random.choice(len(dset), size=300, replace=False))
dset_val = datasets.load_dataset('rotten_tomatoes')['validation']
dset_val = dset_val.select(np.random.choice(len(dset_val), size=300, replace=False))

# fit model
m = AugLinearClassifier(
    checkpoint='textattack/distilbert-base-uncased-rotten-tomatoes',
    ngrams=2, # use bigrams
)
m.fit(dset['text'], dset['label'])

# predict
preds = m.predict(dset_val['text'])
print('acc_val', np.mean(preds == dset_val['label']))

# interpret
print('Total ngram coefficients: ', len(m.coefs_dict_))
print('Most positive ngrams')
for k, v in sorted(m.coefs_dict_.items(), key=lambda item: item[1], reverse=True)[:8]:
    print('\t', k, round(v, 2))
print('Most negative ngrams')
for k, v in sorted(m.coefs_dict_.items(), key=lambda item: item[1])[:8]:
    print('\t', k, round(v, 2))
```



Reference:
```r
@article{singh2023augmenting,
  title={Augmenting interpretable models with large language models during training},
  author={Singh, Chandan and Askari, Armin and Caruana, Rich and Gao, Jianfeng},
  journal={Nature Communications},
  volume={14},
  number={1},
  pages={7913},
  year={2023},
  publisher={Nature Publishing Group UK London}
}
```
