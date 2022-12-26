import re, pickle

from flask import Flask, jsonify
import pandas as pd

app = Flask(__name__)

from flask import request
from flasgger import Swagger, LazyString, LazyJSONEncoder
from flasgger import swag_from
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.models import load_model

import numpy as np


app.json_encoder = LazyJSONEncoder
swagger_template = dict(
    info = {
        'title': LazyString(lambda: 'API Documentation for Data Processing and Modeling'),
        'version': LazyString(lambda: '1.0.0'),
        'description': LazyString(lambda: 'Dokumentasi API untuk Data Processing dan Modeling')
    },
    host = LazyString(lambda: request.host)
)
swagger_config = {
    "headers": [],
    "specs": [
        {
            "endpoint": 'docs',
            "route": '/docs.json'
        }
    ],
    "static_url_path": "/flasgger_static",
    "swagger_ui": True,
    "specs_route": "/docs/"
}
swagger = Swagger(app, template=swagger_template,config=swagger_config)

def cleansing(text):
    text = text.lower()
    text = re.sub('\t[a-zA-Z]*',' ', text)
    text = re.sub('[^a-zA-Z0-9]',' ', text)
    text = re.sub('x[a-z0-9]{1,2}',' ', text)
    text = re.sub('\s+',' ', text)
    return text

def lowerCase(i): 
    return i.lower()

def file_processing(tweet):
    tweet = re.sub(r"rt", "", tweet)
    tweet = re.sub(r"user", "", tweet)
    tweet = re.sub(r'http\S+', '', tweet)
    tweet = re.sub(r"www.\S+", "", tweet)
    tweet = re.sub("@[A-Za-z0-9_]+","", tweet)
    tweet = re.sub("#[A-Za-z0-9_]+","", tweet)
    tweet = re.sub(r'[^\x00-\x7f]',r'', tweet)
    tweet = re.sub(r"[^\w\d\s]+", "", tweet)
    tweet = re.sub(r"x[A-Za-z0-9./]+", "", tweet)
    tweet = re.sub(r'url',' ', tweet)
    return tweet

def clean_file(df):
    df["a"]=df["Tweet"]
    df["a"]=df["a"].apply(lowerCase)
    df["a"]=df["a"].apply(file_processing)
    return df

sentiment = ['negatif', 'netral', 'positif']

file_tokenizer = open('tokenizer.pickle', 'rb')
# max_features = 100000
# tokenizer = Tokenizer(num_words=max_features, split=' ', lower=True)
file_sequencer = open('x_pad_sequences.pickle', 'rb')

load_tokenizer = pickle.load(file_tokenizer)
load_sequencer = pickle.load(file_sequencer)
file_sequencer.close()
model_lstm = load_model('modellstm.h5')
model_cnn = load_model('modelcnn.h5')

def predictFile_LSTM(X_test):
    Y_predict = []
    for index, row in X_test.iterrows():
        sequences_X_test = load_tokenizer.texts_to_sequences([row["Tweet"]])
        X_test_1 = pad_sequences(sequences_X_test, maxlen=load_sequencer.shape[1])
        X_test_1 = np.reshape(X_test_1, (1,load_sequencer.shape[1]))
        result = model_lstm.predict(X_test_1)[0]
        print(result)
        if(np.argmax(result) == 0):
            Y_predict.append("neutral")
        elif (np.argmax(result) == 1):
            Y_predict.append("positive")
        elif (np.argmax(result) == 2):
            Y_predict.append("negative")
    return Y_predict

def predictFile_CNN(X_test):
    Y_predict = []
    for index, row in X_test.iterrows():
        sequences_X_test = load_tokenizer.texts_to_sequences([row["Tweet"]])
        X_test_1 = pad_sequences(sequences_X_test, maxlen=load_sequencer.shape[1])
        X_test_1 = np.reshape(X_test_1, (1,load_sequencer.shape[1]))
        result = model_cnn.predict(X_test_1)[0]
        print(result)
        if(np.argmax(result) == 0):
            Y_predict.append("neutral")
        elif (np.argmax(result) == 1):
            Y_predict.append("positive")
        elif (np.argmax(result) == 2):
            Y_predict.append("negative")
    return Y_predict

@swag_from("docs/lstm.yml", methods=['POST'])
@app.route('/lstm', methods=['POST'])
def lstm():

    text = request.form.get('text')
    cleanse_text = [cleansing(text)]

    # feature = tokenizer.texts_to_sequences(cleanse_text)
    feature = load_tokenizer.texts_to_sequences(cleanse_text)
    feature = pad_sequences(feature, maxlen=load_sequencer.shape[1])

    prediction = model_lstm.predict(feature)
    get_sentiment = sentiment[np.argmax(prediction[0])]

    json_response = {
        'status_code': 200,
        'description': cleanse_text,
        'sentiment': get_sentiment
    }

    response_data = jsonify(json_response)
    return response_data

@swag_from("docs/cnn.yml", methods=['POST'])
@app.route('/cnn', methods=['POST'])
def cnn():

    text = request.form.get('text')
    cleanse_text = [cleansing(text)]

    # feature = tokenizer.texts_to_sequences(cleanse_text)
    feature = load_tokenizer.texts_to_sequences(cleanse_text)
    feature = pad_sequences(feature, maxlen=load_sequencer.shape[1])

    prediction = model_cnn.predict(feature)
    get_sentiment = sentiment[np.argmax(prediction[0])]

    json_response = {
        'status_code': 200,
        'description': cleanse_text,
        'sentiment': get_sentiment
    }

    response_data = jsonify(json_response)
    return response_data

@swag_from("docs/lstm_file.yml", methods=['POST'])
@app.route("/file_lstm/", methods=['POST'])
def file_lstm():
    file = request.files['file']
    df = pd.read_csv(file, encoding=('ISO-8859-1'))
    cleanse_text = clean_file(df[0:10]) 
    cleanse_text['sentiment'] = predictFile_LSTM(cleanse_text)
    cleanse_text.to_csv('sentiment lstm.csv', index=False, header=False)

    return jsonify({"sentiment" : "Berhasil dibuat file"})

@swag_from("docs/cnn_file.yml", methods=['POST'])
@app.route("/file_cnn/", methods=['POST'])
def file_cnn():
    file = request.files['file']
    df = pd.read_csv(file, encoding=('ISO-8859-1'))
    cleanse_text = clean_file(df[0:10]) 
    cleanse_text['sentiment'] = predictFile_CNN(cleanse_text)
    cleanse_text.to_csv('sentiment cnn.csv', index=False, header=False)

    return jsonify({"sentiment" : "Berhasil dibuat file"})

if __name__ == '__main__':
    app.run(debug=True)