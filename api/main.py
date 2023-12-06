from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from rakun2 import RakunKeyphraseDetector

app = FastAPI()

default_hyperparameters = {
    "num_keywords": 10,
    "merge_threshold": 1.1,
    "alpha": 0.3,
    "token_prune_len": 3,
}

class TextInput(BaseModel):
    text: str
    hyperparameters: dict = default_hyperparameters

class PDFInput(BaseModel):
    file: UploadFile
    hyperparameters: dict = default_hyperparameters

@app.post("/get_keywords/", response_model=dict)
async def get_keywords(input_data: TextInput):
    hyperparameters = input_data.hyperparameters
    keyword_detector = RakunKeyphraseDetector(hyperparameters)
    keywords = keyword_detector.find_keywords(input_data.text, input_type="string")
    return {"keywords": keywords}

