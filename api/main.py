import os
import shutil
from typing import Dict

from fastapi import FastAPI, File, UploadFile
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

@app.post("/get_keywords/", response_model=Dict[str, list])
async def get_keywords(input_data: TextInput):
    hyperparameters = input_data.hyperparameters
    keyword_detector = RakunKeyphraseDetector(hyperparameters)
    keywords = keyword_detector.find_keywords(input_data.text, input_type="string")
    return {"keywords": keywords}

@app.post("/get_keywords_pdf/", response_model=Dict[str, list])
async def get_keywords(file: UploadFile = File(...), hyperparameters: dict = default_hyperparameters):
    upload_directory = "uploaded_files"
    os.makedirs(upload_directory, exist_ok=True)

    file_location = os.path.join(upload_directory, file.filename)

    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    keyword_detector = RakunKeyphraseDetector(hyperparameters)
    keywords = keyword_detector.find_keywords(file_location, input_type="pdf")

    os.remove(file_location)

    return {"keywords": keywords}