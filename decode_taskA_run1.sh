import argparse
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import RandomOverSampler

import string
import re
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
stop_words = stopwords.words('english')

def preprocess_text(text):
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Remove digits
    text = re.sub(r'\d+', '', text)
    # Convert to lowercase
    text = text.lower()
    # Remove stop words
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

def main(test_file):
    val_df = pd.read_csv(test_file)
    val_df['dialogue'] = val_df['dialogue'].apply(preprocess_text)

    tfidf = TfidfVectorizer(max_features=10000)

    X_val = tfidf.transform(val_df['dialogue']).toarray()
    y_val = val_df['section_header']

    svc_model = Pipeline([('svc', LinearSVC())])
    svc_model.fit(X_train_resampled, y_train_resampled)
    y_pred = svc_model.predict(X_val)

    # Write the output to a CSV file
    output_df = pd.DataFrame({'section_header': y_pred})
    output_df.to_csv('output.csv', index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('test_file', help='Path to the test CSV file')
    args = parser.parse_args()

    main(args.test_file)
