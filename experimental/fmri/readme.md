Code in this folder copied from https://github.com/HuthLab/deep-fMRI-dataset. See that wonderful repo for up-to-date code!

# deep-fMRI-dataset
Code accompanying data release of natural language listening data from 5 fMRI sessions for each of 8 subjects (LeBel et al.) that can be found at [openneuro](https://openneuro.org/datasets/ds003020).

- need to grab `em_data` directory from there
- need to download data following the below instructions below
- need to set appropriate paths in encoding/feature_space.py
- download data with `python 00_load_dataset.py`
    - This function will create a `data` dir if it does not exist and will use [datalad](https://github.com/datalad/datalad) to download the preprocessed data as well as feature spaces needed for fitting [semantic encoding models](https://www.nature.com/articles/nature17637). It will download ~20gb of data. 

# model fitting
- `python 01_fit_encoding.py --subject UTS03 --feature eng1000`
    - The other optional parameters that encoding.py takes such as sessions, ndelays, single_alpha allow the user to change the amount of data and regularization aspects of the linear regression used. 
    - This function will then save model performance metrics and model weights as numpy arrays. 