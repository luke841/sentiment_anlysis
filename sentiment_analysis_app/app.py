from flask import Flask, request, jsonify, render_template

import pandas as pd
from pathlib import Path
import joblib

# Perhaps put this in a configurations file
PROJECT_ROOT_FOLDER = Path(__file__).resolve().parent


# note determine how to get the projects base path
with open(PROJECT_ROOT_FOLDER / 'model.joblib', 'rb') as f:
    model = joblib.load(f)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():

    input_text = request.form['text']
    predicted_sentiment = model.predict(pd.Series([input_text]))[0]
    if predicted_sentiment == 1:
        output = 'positive'
    else:
        output = 'negative'

    return render_template(
        'index.html',
        sentiment=f'Predicted sentiment of "{input_text}" is {output}.'
    )


if __name__ == "__main__":
    app.run(debug=True)
