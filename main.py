from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

class TweetRequest(BaseModel):
    text: str
    
class ClassifyResponse(BaseModel):
    label: str
    score: float
    
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
    res = dict()
    res["label"] = "undefined"
    res["score"] = 0.99921
    return res

@app.post(
    "/hate-speech-topic/predict", 
    response_model=ClassifyResponse,
    tags=["Hate Speech"], 
    name="post_single_hate_speech_topic_classifier", 
    summary="Classify hate speech with topic from single text",
    description="Classify hate speech with topic from single text"
)
def hate_speech_topic_classifier(request: TweetRequest):
    res = dict()
    res["label"] = "undefined"
    res["score"] = 0.99921
    return res

