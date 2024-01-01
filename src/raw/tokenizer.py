from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from dataset import df

df = pd.read_csv('pre_preprocessed_dataset.csv')
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(df['processed_text'])



