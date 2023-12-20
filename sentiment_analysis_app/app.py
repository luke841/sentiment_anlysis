from flask import Flask, request, jsonify, render_template
import pandas as pd
import joblib


app = Flask(__name__)
with open('sentiment_analysis_app/model.joblib', 'rb') as f:
    model = joblib.load(f)


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
