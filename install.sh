

# create a new conda environment
conda create -y --name SSNdhanyadivyakavitha_taskA_venv python=3.8

# activate the environment
source activate SSNdhanyadivyakavitha_taskA_venv


# Upgrade pip and install necessary packages
pip install --upgrade pip
pip install pandas numpy nltk scikit-learn imbalanced-learn

# Download nltk data
python -c "import nltk; nltk.download('wordnet')"


# Deactivate the virtual environment
deactivate
