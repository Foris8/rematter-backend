from fastapi import FastAPI, File, UploadFile
import shutil
from easyocr import Reader
import tempfile
import os
import requests
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import json

# Define the FastAPI app
app = FastAPI()
origins = ["*"]
load_dotenv()
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

reader = Reader(lang_list=['en'])

# Your OpenAI API key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


def ask_openai(context):
    prompt = f"""Given the details from a driver's license, answer the following questions in a hash format where each question's answer is provided in the specified format (name, short format, date format, numeric format, or a single word as appropriate):

    1. What is the full name on the driver's license? (name format)
    2. Provide the address from the driver's license. (short format)
    3. State the date of birth on the driver's license. (date format)
    4. When was the driver's license issued? (date format)
    5. What is the expiration date of the driver's license? (date format)
    6. What is the driver's license number? (numeric format)
    7. What is the sex indicated on the driver's license? (single word)

    Please format the answers like this:
    {{
    "Full Name": "Answer",
    "Address": "Answer",
    "Date of Birth": "Answer",
    "Issued Date": "Answer",
    "Expiration Date": "Answer",
    "License Number": "Answer",
    "Sex": "Answer"
    }}

    Context: {context}
    """

    response = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {OPENAI_API_KEY}"
        },
        json={
            "model": "gpt-3.5-turbo",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
        }
    )

    response_data = response.json()
    print(response_data)
    answer_text = response_data.get("choices", [{}])[0].get(
        "message", {"content": ""}).get("content", "").strip()

    # Parsing logic for the structured answer might be needed here,
    # depending on how closely the model follows the instruction.

    return answer_text




@app.get("/test")
def test():
    return {"message": "The FastAPI application is running successfully!"}


@app.post("/reader")
async def upload_file(file: UploadFile = File(...)):
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
        slope_ths=0.4,
        width_ths=0.7
    )
    
    ocr_text = " ".join(result)
    os.remove(temp_file_path)

    details = ask_openai(ocr_text)
    
    details_dict = json.loads(details)

    formatted_details = {key.lower().replace(" ", "_"): value for key, value in details_dict.items()}

    return formatted_details
