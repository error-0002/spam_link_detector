# preprocess.py
import re
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import os

STOP = set(stopwords.words('english'))
ps = PorterStemmer()

def clean_text(s):
    if pd.isna(s): return ""
    s = str(s).lower()
    s = re.sub(r'http\S+|www\.\S+|https\S+', ' ', s)  # remove urls
    s = re.sub(r'[^a-z\s]', ' ', s)                   # keep letters
    s = re.sub(r'\s+', ' ', s).strip()
    tokens = [ps.stem(w) for w in s.split() if w not in STOP]
    
    return " ".join(tokens)

def run():
    os.makedirs("data", exist_ok=True)
    df = pd.read_csv("data/sms_spam.tsv", sep='\t', header=None, names=['label', 'message'])
    print("Loaded:", df.shape)
    df = df.drop_duplicates(subset='message')
    df['message_clean'] = df['message'].apply(clean_text)
    df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})
    df = df[df['message_clean'].str.strip() != ""]
    df.to_csv("data/sms_clean.csv", index=False)
    print("Cleaned data saved to data/sms_clean.csv")

if __name__ == "__main__":
    run()
