from wildtime import dataloader
import argparse
from wildtime.configs.eval_stream import configs_huffpost


"""
Due to conflicting package dependencies between imodelsx and wildtime, 
we download the HuffPost dataset separately by creating a python environment 
and installing the wildtime package and not the imodelsx package. 
Once we have downloaded the data and stored it locally, we will create 
a new environment to train the emb-gam model
"""

config = argparse.Namespace(**configs_huffpost.configs_huffpost_agem)
data = dataloader.getdata(config)
