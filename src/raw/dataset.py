# Loading and Pre-Processing the dataset

import spacy
import pandas as pd
import regex

nlp = spacy.load('en_core_web_sm')

def pre_process(text):

    text = text.lower()

    # remove single characters
    text = regex.sub(r'\b[A-Za-z]\b', '', text)

    # remove whitespace
    text = regex.sub(r'\s+', ' ', text)

    # remove stand-alone numbers
    # text = regex.sub(r'\b\d+\b', '', text)

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

df['processed_text'] = df['text'].apply(pre_process)
df.to_csv('pre_preprocessed_dataset.csv')
print("done")

output_file = "/home/prashant/Downloads/llm-detect-ai-generated-text/pre-processed111.csv"
df.to_csv(output_file, index=False)
