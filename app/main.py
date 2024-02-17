from easyocr import Reader
from transformers import pipeline



reader = Reader(lang_list=['en'])
result = reader.readtext(
    image='app/annotated_image_with_labels.jpg',
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

print(result)


# Join the OCR results into a single text block
ocr_text = ",".join(result)
print("OCR Text:", ocr_text)

questions = [
    "What is the user's full name as displayed on the driver's license?",
    "What is the complete address listed on the driver's license?",
    "What is the date of birth?",
    "What is the issuance date of the driver's license?",
    "What is the expiration date of the driver's license?",
    "What is the license number?",
    "what is the sex?"
    # Add more questions as needed
]

# Load the question answering pipeline with BERT
qa_pipeline = pipeline(
    "question-answering", model="bert-large-uncased-whole-word-masking-finetuned-squad")

details = {}

for question in questions:
    input = {
        'question': question,
        'context': ocr_text
    }
    answer = qa_pipeline(input)
    details[question] = answer['answer']

# Print extracted details
for detail, answer in details.items():
    print(f"{detail}: {answer}")
