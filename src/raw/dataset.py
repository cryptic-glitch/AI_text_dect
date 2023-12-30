# Loading the dataset and pre-processing the data

import string
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
# Pre-Processing the data

def pre_process(text):

    #  lower case
    text = text.lower()

    #  Remove puntuation
    text = ''.join([char for char in text if char not in string.punctuation])

    #  Tokenization
    tokens = word_tokenize(text)

    # Stop words like -> and, the etc
    stop_words = set(stopwords.words('english'))
    tokens = ' ' .join([word for word in text.split() if word.lower() not in stop_words])

    # Lemmatization or Stemming
    lemmatizer = WordNetLemmatizer()
    tokens = ' '.join([lemmatizer.lemmatize(word) for word in text.split()])

    #  Joining
    processed_text = " ".join(tokens)

    return processed_text

abs_file_path = "/home/prashant/Downloads/llm-detect-ai-generated-text/train_essays.csv"
df = pd.read_csv(abs_file_path)

df['processed_text'] = df['text'].apply(pre_process)

# output_file = "/home/prashant/Downloads/llm-detect-ai-generated-text/pre-processed.csv"
# df.to_csv(output_file, index=False)
# print("Done")
