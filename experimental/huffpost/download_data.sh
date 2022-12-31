#!/bin/bash
# download huffpost data
FILE="Data/huffpost.pkl"
if [ ! -f "$FILE" ]; then
    echo "Downloading $FILE"
    conda create --env huffpost_download_env
    conda activate huffpost_download_env
    pip install wildtime
    python download_huffpost.py
    conda decative huffpost_download_env
    conda env remove -n huffpost_download_env
fi






