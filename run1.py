import argparse
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import RandomOverSampler

parser = argparse.ArgumentParser()
parser.add_argument('test_file', help='Path to the test CSV file')
args = parser.parse_args()

test_path = args.test_file

train_path = '/workspaces/SSNdhanyadivyakavitha/TaskA-TrainingSet.csv'
val_path = '/workspaces/SSNdhanyadivyakavitha/TaskA-ValidationSet.csv'

train_df = pd.read_csv(train_path)
val_df = pd.read_csv(val_path)
test_df = pd.read_csv(test_path)

import string
import re
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')

stop_words = stopwords.words('english')

def preprocess_text(text):
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Remove digits
    text = re.sub(r'\d+', '', text)
    # Convert to lowercase
    text = text.lower()
    # Remove stop words
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

train_df['dialogue'] = train_df['dialogue'].apply(preprocess_text)
val_df['dialogue'] = val_df['dialogue'].apply(preprocess_text)
test_df['dialogue'] = test_df['dialogue'].apply(preprocess_text)

tfidf = TfidfVectorizer(max_features=10000)

X_train = tfidf.fit_transform(train_df['dialogue']).toarray()
y_train = train_df['section_header']

X_val = tfidf.transform(val_df['dialogue']).toarray()
y_val = val_df['section_header']

X_test = tfidf.transform(test_df['dialogue']).toarray()

ros = RandomOverSampler()
X_train_resampled, y_train_resampled = ros.fit_resample(X_train, y_train)

svc_model = Pipeline([('svc', LinearSVC())])
svc_model.fit(X_train_resampled, y_train_resampled)

y_pred = svc_model.predict(X_test)

output_path = 'TaskA_predictions.csv'
test_df['SystemOutput'] = y_pred
test_df = test_df.rename(columns={'id':'TestID', 'SystemOutput':'section_header'}).drop(['dialogue'], axis=1)
test_df.to_csv(output_path, index=False)

print("Predictions saved to", output_path)
