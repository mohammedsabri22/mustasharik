from datetime import datetime
from django.shortcuts import render
from django.http import JsonResponse,HttpRequest
from django.views.decorators.csrf import csrf_exempt
from openai import OpenAI

import json
import logging
from PIL import Image 
from pytesseract import pytesseract 
import fitz
import re


logger = logging.getLogger(__name__)

client = OpenAI(
    api_key = "key-api"
)

def home(request):
    return render( request, 'app/index.html' )



def chat_view(request):

  if request.method == 'POST':
    user_message = request.POST.get('message')

    if not user_message:
      return JsonResponse({'error': 'No message provided'}, status=400)

    gpt_response = chat_with_gpt(user_message) 
    return JsonResponse({'response': gpt_response})

  return render(request, 'app/chat_file.html') 



def chat_with_gpt(user_message):

  try:
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",  # Use the appropriate GPT model
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": user_message}
        ]
    )

    gpt_response = response.choices[0].message.content
    return gpt_response

  except Exception as e:
    print(f"Error calling OpenAI API: {e}")
    return f"An error occurred: {e}"


@csrf_exempt
def contract(request):
    if request.method == 'POST':
        contract_file = request.FILES.get('contract-file')
        contract_type = request.POST.get('contract-type')
        
        if contract_file:
            extracted_text = extract_text(contract_file)  # Extract text
            return JsonResponse({'extracted_text': extracted_text, 'contract_type': contract_type})
        else:
            return JsonResponse({'error': 'No file uploaded'}, status=400)

    return render(request, 'app/contract.html')


def extract_text(file):
    if file.name.endswith('.pdf') :
        return pdf_to_string(file)
    else:
        return image_to_string(file)


def pdf_to_string(pdf_file):
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text


def image_to_string(image_file):
    image = Image.open(image_file)
    text = pytesseract.image_to_string(image, lang='ara')  # 'ara' for Arabic
    return text

def extract_terms_and_conditions(text):
    # Define keywords to identify the Terms and Conditions section
    keywords = ["الشروط والأحكام", "البنود التالية","ماياتي"]
    
    # Find the start and end of the section
    start = None
    for keyword in keywords:
        start = text.find(keyword)
        if start != -1:
            break
    
    if start == -1:
        return None  # Section not found
    
    # Extract the section (you may need to adjust this logic based on your contract structure)
    section = text[start:]
    return section

def split_into_clauses(section):
    # Split the section into clauses based on punctuation
    clauses = []
    for line in section.split('\n'):
        # Split by periods or Arabic equivalents
        line_clauses = [clause.strip() for clause in re.split(r'[\.۔]', line) if clause.strip()]
        clauses.extend(line_clauses)
    
    return clauses
