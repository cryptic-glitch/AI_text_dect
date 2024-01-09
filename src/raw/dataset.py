# Loading and Pre-Processing the dataset

import spacy
import pandas as pd
import regex
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
from tqdm import tqdm

nlp = spacy.load('en_core_web_sm')

def pre_process(text):
    text = text.lower()

    # remove single characters
    text = regex.sub(r'\b[A-Za-z]\b', '', text)

    # remove whitespace
    text = regex.sub(r'\s+', ' ', text)

    # remove repeated characters
    text = regex.sub(r'\b(\w)\1+\b', r'\1\1', text)

    # remove words with 1 or 2 characters
    text = regex.sub(r'\b(?:\w{1,2})\b', '', text)

    doc = nlp(text)
    tokens = [token.lemma_.lower() for token in doc if not token.is_punct and not token.is_stop]

    # Join tokens back to text
    processed_text = ' '.join(tokens)

    return processed_text


abs_file_path = "/home/prashant/Downloads/llm-detect-ai-generated-text/train_essays.csv"
df = pd.read_csv(abs_file_path)

tqdm.pandas(desc="Pre-processing")
df['processed_text'] = df['text'].progress_apply(pre_process)
df.to_csv('pre_preprocessed_dataset.csv')
print("Pre-processing done")

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(tqdm(df['processed_text'], desc="TF-IDF Vectorization"))

joblib.dump(X_train_tfidf, 'X_train_tfidf.pkl')
print("TF-IDF Vectorization done")
