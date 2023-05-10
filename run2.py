import argparse
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.utils import resample
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('test_data', type=str, help='Path to test data CSV file')
args = parser.parse_args()

# Read the training and validation datasets
train_path = 'TaskA-TrainingSet.csv'
val_path = 'TaskA-TrainingSet.csv'

train_df = pd.read_csv(train_path)
val_df = pd.read_csv(val_path)

# Balance the dataset using oversampling
max_samples = train_df['section_header'].value_counts().max()
train_balanced = pd.DataFrame(columns=train_df.columns)
for label in train_df['section_header'].unique():
    df_label = train_df[train_df['section_header'] == label]
    df_oversampled = resample(df_label, replace=True, n_samples=max_samples, random_state=42)
    train_balanced = pd.concat([train_balanced, df_oversampled])

# Preprocess the text data using lemmatization
import nltk
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

def lemmatize(text):
    words = text.split()
    return ' '.join([lemmatizer.lemmatize(word) for word in words])

train_balanced['lemmatized_text'] = train_balanced['dialogue'].apply(lemmatize)
val_df['lemmatized_text'] = val_df['dialogue'].apply(lemmatize)

# Encode the target labels
le = LabelEncoder()
train_balanced['label'] = le.fit_transform(train_balanced['section_header'])
val_df['label'] = le.transform(val_df['section_header'])

# Define the pipeline
pipeline = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('classifier', LogisticRegression(penalty='l2', solver='lbfgs', max_iter=1000, random_state=42))
])

# Train the model
pipeline.fit(train_balanced['lemmatized_text'], train_balanced['label'])

# Load the test data
test_df = pd.read_csv(args.test_data)

# Preprocess the text data using lemmatization
test_df['lemmatized_text'] = test_df['dialogue'].apply(lemmatize)

# Make predictions
test_df['SystemOutput'] = le.inverse_transform(pipeline.predict(test_df['lemmatized_text']))

# Rename columns and drop the 'dialogue' column
test_df = test_df.rename(columns={'ID': 'TestID', 'section_header': 'SystemOutput'})
test_df = test_df.drop(columns=['lemmatized_text'])

# Save the output to a CSV file
test_df.to_csv('TaskA_predictions_run2.csv', index=False)
