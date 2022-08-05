from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__)
cors = CORS(app)

@app.route("/", methods=["GET"])
def main():
    response = jsonify({'message': 'Please user /predict or /predict-multi with text as request.'})
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response

@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        response = jsonify({'label': "none"})
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response
        
@app.route("/predict-multi", methods=["POST"])
def predictx():
    if request.method == "POST":
        response = jsonify({'label': "none"})
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response

