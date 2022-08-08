import dill

import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

dill._dill._reverse_typemap['ClassType'] = type

class TweetRequest(BaseModel):
    text: str
    
class ClassifyResponse(BaseModel):
    label: str
    score: float
    
class ClassifyResponseLite(BaseModel):
    label: str
    
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post(
    "/hate-speech/predict", 
    response_model=ClassifyResponse,
    tags=["Hate Speech"], 
    name="post_single_hate_speech_classifier", 
    summary="Classify hate speech from single text",
    description="Classify hate speech from single text"
)
def hate_speech_classifier(request: TweetRequest):
    text = request.text
    
    file =  open('model/hatespeech.pkl', 'rb')
    model = dill.load(file)
    file.close()
    
    pred_proba = model.predict_proba(pd.DataFrame({'text': [text]}))[0]
    
    res = dict()
    if (pred_proba >= 0.5):
        res["label"] = "non hatespeech"
        res["score"] = int(pred_proba * 100)
    else:      
        res["label"] = "hatespeech"
        res["score"] = 100 - int(pred_proba * 100)
    
    return res

@app.post(
    "/hate-speech-topic/predict", 
    response_model=ClassifyResponseLite,
    tags=["Hate Speech"], 
    name="post_single_hate_speech_topic_classifier", 
    summary="Classify hate speech with topic from single text",
    description="Classify hate speech with topic from single text"
)
def hate_speech_topic_classifier(request: TweetRequest):
    text = request.text
    
    file =  open('model/hatespeech_topic.pkl', 'rb')
    model = dill.load(file)
    file.close()
    
    pred = model.predict(pd.DataFrame({'text': [text]}))[0]
    
    classes = ['gender hatespeech', 'other hatespeech', 'physical hatespeech', 'race hatespeech', 'religion hatespeech']
    
    res = dict()
    res["label"] = classes[pred]
    
    return res

