from fastapi import FastAPI, File, UploadFile
import shutil
from easyocr import Reader
from transformers import pipeline
import tempfile
import os
from fastapi.middleware.cors import CORSMiddleware

# define the fastapi
app = FastAPI()
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

reader = Reader(lang_list=['en'])
qa_pipeline = pipeline(
    "question-answering", model="bert-large-uncased-whole-word-masking-finetuned-squad")


@app.get("/test")
def test():
    return {"message": "The FastAPI application is running successfully!"}

@app.post("/reader")
def upload_file(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        shutil.copyfileobj(file.file, temp_file)
        temp_file_path = temp_file.name

    result = reader.readtext(
        image=temp_file_path,
        detail=0,
        paragraph=False,
        allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789/-,'? \" \' ",
        contrast_ths=0.4,
        adjust_contrast=0.6,
        text_threshold=0.8,
        low_text=0.4,
        link_threshold=0.4,
        canvas_size=2560,
        mag_ratio=1.0,
        slope_ths= 0.4,
        width_ths = 0.7

    )



    # Join the OCR results into a single text block
    ocr_text = ",".join(result)
    os.remove(temp_file_path)
    
    questions = [
        "What is the user's full name as displayed on the driver's license?",
        "What is the complete address listed on the driver's license?",
        "What is the date of birth?",
        "What is the issued date of the driver's license?",
        "What is the expiration date of the driver's license?",
        "What is the license number?",
        "what is the sex?"
    ]

    details = {}
    answers = [qa_pipeline({'question': q, 'context': ocr_text})[
        'answer'] for q in questions]

    details = {
        "name": answers[0],
        "address": answers[1],
        "DOB": answers[2],
        "issued date": answers[3],
        "expired date": answers[4],
        "license number": answers[5],
        "sex": answers[6]
    }


    return {"details": details} 
