import dill

import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

dill._dill._reverse_typemap['ClassType'] = type

class TweetRequest(BaseModel):
    text: str
    
class ClassifyResponseLite(BaseModel):
    label: str
    
class ClassifyResponse(BaseModel):
    label: str
    score: float

class ClassifyResponseExtra(BaseModel):
    label: str
    score: float
    category: str
    
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

file =  open('model/hatespeech.pkl', 'rb')
model_hs = dill.load(file)
file.close()

file =  open('model/hatespeech_topic.pkl', 'rb')
model_hs_topic = dill.load(file)
file.close()

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
    
    pred_proba = model_hs.predict_proba(pd.DataFrame({'text': [text]}))[0]
    
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
    
    pred = model_hs_topic.predict(pd.DataFrame({'text': [text]}))[0]
    
    classes = ['gender hatespeech', 'other hatespeech', 'physical hatespeech', 'race hatespeech', 'religion hatespeech']
    
    res = dict()
    res["label"] = classes[pred]
    
    return res

@app.post(
    "/hate-speech-pipeline/predict", 
    response_model=ClassifyResponseExtra,
    tags=["Hate Speech"], 
    name="post_single_hate_speech_pipeline_classifier", 
    summary="Classify hate speech with pipeline from single text",
    description="Classify hate speech with pipeline from single text"
)
def hate_speech_pipeline_classifier(request: TweetRequest):
    text = request.text
    
    pred_proba = model_hs.predict_proba(pd.DataFrame({'text': [text]}))[0]
    
    res = dict()
    if (pred_proba >= 0.5):
        res["label"] = "non hatespeech"
        res["score"] = int(pred_proba * 100)
        res["category"] = "-"
    else:      
        res["label"] = "hatespeech"
        res["score"] = 100 - int(pred_proba * 100)
        
        pred = model_hs_topic.predict(pd.DataFrame({'text': [text]}))[0]
        
        classes = ['gender hatespeech', 'other hatespeech', 'physical hatespeech', 'race hatespeech', 'religion hatespeech']
        
        res["category"] = classes[pred]
    
    return res
