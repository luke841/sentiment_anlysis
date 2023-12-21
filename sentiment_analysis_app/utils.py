#Import neccessory libraries
import numpy as np
import pandas as pd
import regex as re
import joblib
from sklearn.base import BaseEstimator, TransformerMixin
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


# Downlaod the stop words
nltk.download('stopwords')

# Define stop words
stop_words_keywords = set(stopwords.words('english'))

# Add additional stop words for keyword extraction
additional_stop_words = [
    "will", "always", "go", "one", "very", "good", "only", "mr", "lot", "two",
    "th", "etc", "don", "due", "didn", "since", "nt", "ms", "ok", "almost",
    "put", "pm", "hyatt", "grand", "till", "add", "let", "hotel", "able",
    "per", "st", "couldn", "yet", "par", "hi", "well", "would", "I", "the",
    "s", "also", "great", "get", "like", "take", "thank"
]

stop_words_keywords.update(additional_stop_words)


class TextPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, stop_words=None):
        self.stop_words = stop_words or set()

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.apply(self.clean_text)

    def clean_text(self, text):
        if not isinstance(text, str):
            return ''

        # Custom text cleaning and preprocessing steps
        text = re.sub(r'\d+', '', text)
        text = text.translate(str.maketrans("", "", string.punctuation))
        text = text.lower()
        words = word_tokenize(text)
        words = [word for word in words if word not in self.stop_words]
        lemmatizer = WordNetLemmatizer()
        words = [lemmatizer.lemmatize(word) for word in words]
        cleaned_text = ' '.join(words)

        return cleaned_text
