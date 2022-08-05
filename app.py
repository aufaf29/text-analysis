from typing import Union
from fastapi import FastAPI

import uvicorn

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}

if __name__ == '__main__':
    uvicorn.run('app:app', host='0.0.0.0', port=5000)

# from flask import Flask, jsonify, request
# from flask_cors import CORS

# app = Flask(__name__)
# cors = CORS(app)

# @app.route("/", methods=["GET"])
# def main():
#     response = jsonify({'message': 'Please user /predict or /predict-multi with text as request.'})
#     response.headers.add('Access-Control-Allow-Origin', '*')
#     return response

# @app.route("/predict", methods=["POST"])
# def predict():
#     if request.method == "POST":
#         response = jsonify({'label': "none"})
#         response.headers.add('Access-Control-Allow-Origin', '*')
#         return response
        
# @app.route("/predict-multi", methods=["POST"])
# def predictx():
#     if request.method == "POST":
#         response = jsonify({'label': "none"})
#         response.headers.add('Access-Control-Allow-Origin', '*')
#         return response

