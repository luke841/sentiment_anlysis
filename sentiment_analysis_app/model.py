# This file creates the 'pipe' NLP model and saves it as model.joblib

# Import libraries
import pandas as pd
import joblib

from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from utils import TextPreprocessor
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

if __name__ == "__main__":
   #may need to change the following to your location of sentiments.csv
   df = pd.read_csv(r'C:\Users\bhara\Downloads\Hotel_sentiment _project\google_reviews3.csv') 
    # Fill or drop null values
df['text'].fillna('', inplace=True)  # Replace 'your_column' with the actual column name
# Create binary labels
df['label'] = df['rating'].apply(lambda x: 1 if x >= 4 else 0)
pipeline = ImbPipeline([
    ('tfidf', TfidfVectorizer(stop_words='english')),  # TF-IDF Vectorization
    ('smote', SMOTE(random_state=42)),  # SMOTE for oversampling
    ('classifier', LogisticRegression(C=100, max_iter=1000, penalty='l2', solver='saga', fit_intercept=False))  # Logistic Regression
])

print("processing")

# Train the model using the pipeline
pipeline.fit(df['text'],df['label'])
joblib.dump(pipeline, open('model.joblib','wb'))

