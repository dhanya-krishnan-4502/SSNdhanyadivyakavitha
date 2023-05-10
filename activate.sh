#!/bin/bash

# Create virtual environment
python3 -m venv SSNdhanyadivyakavitha_taskA_venv

# Activate virtual environment
source SSNdhanyadivyakavitha_taskA_venv/bin/activate

# Install required packages
pip install pandas numpy scikit-learn imbalanced-learn nltk

# install specific versions of scikit-learn and joblib
pip install -U scikit-learn==0.24.2 joblib==1.0.1

# download NLTK data
python -m nltk.downloader wordnet
