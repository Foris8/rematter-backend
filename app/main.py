from fastapi import FastAPI, File, UploadFile, HTTPException
import shutil
import tempfile
import os
import requests
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import json
from starlette.responses import Response
from starlette.responses import JSONResponse  # Import JSONResponse
import base64

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

# reader = Reader(lang_list=['en'])

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


def encode_image_to_base64(image_path):
    """Encodes the image at the given path to base64."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')
    

def process_image_with_openai(base64_image):
    """Sends the base64 encoded image to the OpenAI API for analysis."""
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }
    
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
    "full_name": "Answer",
    "address": "Answer",
    "date_of_birth": "Answer",
    "issued_date": "Answer",
    "expiration_date": "Answer",
    "license_number": "Answer",
    "sex": "Answer"
    }}"""

    payload = {
        "model": "gpt-4-vision-preview",  # Hypothetical model
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
        "max_tokens": 2000
    }

    response = requests.post(
        "https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    if response.status_code == 200:
        
        return response.json()
    else:
        return {"error": "Failed to analyze image", "details": response.text}
    

@app.get("/test")
def test():
    return {"message": "The FastAPI application is running successfully!"}


@app.post("/reader")
def upload_file(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        shutil.copyfileobj(file.file, temp_file)
        temp_file_path = temp_file.name #set the file_path
    
    # result = reader.readtext(
    #     image=temp_file_path,
    #     detail=0,
    #     paragraph=False,
    #     allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789/-,'? \" \' ",
    #     contrast_ths=0.4,
    #     adjust_contrast=0.6,
    #     text_threshold=0.8,
    #     low_text=0.4,
    #     link_threshold=0.4,
    #     canvas_size=2560,
    #     mag_ratio=1.0,
    #     slope_ths=0.4,
    #     width_ths=0.7
    # )
    
    # ocr_text = " ".join(result)
    # details = process_image_with_openai(temp_file_path)
    # os.remove(temp_file_path)

    base64_image = encode_image_to_base64(temp_file_path)
    analysis_result = process_image_with_openai(base64_image)
    print(analysis_result)
    if "choices" in analysis_result and analysis_result["choices"]:
        content_str = analysis_result["choices"][0]["message"]["content"]

        try:
            # Parse the JSON string into a Python dictionary
            content_dict = json.loads(content_str)

            # Now you can use content_dict as a normal Python dictionary
            # For example, to send it back as a JSON response:
            return JSONResponse(content=content_dict)

        except json.JSONDecodeError as e:
            # Handle JSON parsing error (e.g., logging, return an error response)
            return HTTPException(status_code=400, detail=f"Failed to decode JSON content: {str(e)}")
    else:
        # Handle the case where the expected data is not in the response
        return HTTPException(status_code=400, detail="Invalid response from the analysis API")
