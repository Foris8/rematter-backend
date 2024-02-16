from fuzzywuzzy import process
import easyocr
import re
import spacy
from textblob import TextBlob
from spacy.tokens import Span

# Load the spaCy model
nlp = spacy.load("en_core_web_sm")

from easyocr import Reader

reader = Reader(lang_list=['en'])
result = reader.readtext(
    image='app/2.jpeg',
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

result = ' '.join(result)
doc = nlp(result)

extracted_info = {
    "Names": [],
    "Dates": [],
    "ID": None,
    "Address": None
}

for ent in doc.ents:
    if ent.label_ in ["PERSON"]:
        extracted_info["Names"].append(ent.text)
    elif ent.label_ in ["DATE"]:
        extracted_info["Dates"].append(ent.text)

id_match = re.search(r'\b\d{3} \d{3} \d{3}\b', result)
if id_match:
    extracted_info["ID"] = id_match.group()

# Starting point for the address, based on the known start
address_start = result.find("204 W")
# End point, adjust based on your data
address_end = result.find("NY 10025") + len("NY 10025")
if address_start != -1 and address_end != -1:
    extracted_info["Address"] = result[address_start:address_end]

# Print extracted information
print("Extracted Information:")
for key, value in extracted_info.items():
    print(f"{key}: {value}")
