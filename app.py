"""
Description: Flask app for Movie review sentimental analysis

@author: Kishorlal
"""

import pickle
import re
import os

# Importing Keras library
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Importing Flask library
from flask import Flask, request, render_template

# Create flask object
app = Flask(__name__, template_folder=os.path.join(os.path.dirname(__file__),r"templates"))

# Saved path of our model
MODEL_PATH = os.path.join(os.path.dirname(__file__),r"SentimentalAnalysis_LSTM.h5")

# Load your trained model
model = load_model(MODEL_PATH)

#Load tokenizer
tokenizerFilePath=os.path.join(os.path.dirname(__file__),r"tokenizer.pickle")
data=open(tokenizerFilePath, 'rb')
tokenizer = pickle.load(data)

@app.route('/')
def home():
	return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
    	# Get the file from post request
        message = request.form['message']
        myInput=re.sub(r'<[^<>]+>', repl=" ",string=message)
        myInput=re.sub(r'[^a-zA-Z0-9\s]', repl=" ",string=myInput)
        preprocess=tokenizer.texts_to_sequences([myInput])
        preprocess=pad_sequences(preprocess,maxlen=500)
        prediction = model.predict(preprocess)
        prediction=prediction[0][1]
        return render_template('result.html', prediction=prediction)
        
    return None

if __name__ == '__main__':
	app.run(debug=True)