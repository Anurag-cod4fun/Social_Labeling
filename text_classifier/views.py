from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt


# Create your views here.

import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import LancasterStemmer,WordNetLemmatizer,PorterStemmer,RegexpStemmer,SnowballStemmer
from nltk.util import ngrams
from contractions import fix
from unidecode import unidecode
from string import punctuation

from wordcloud import WordCloud

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

def remove_blank(text):
    if text is not None:
        text_data = text.replace("\n", "").replace("\t", "")
        return text_data
    else:
        return "The text is empty"

def expanding_text(text):
    text_data = fix(text)
    return text_data

stopword_list = stopwords.words("english")

def handle_accented_chr(text):
    text_data = unidecode(text)
    return text_data

def clean_text(text):
    text_data = text.lower()
    tokens = word_tokenize(text_data)
    clean_data = [i for i in tokens if i not in punctuation]
    clean_data = [i for i in clean_data if i not in stopword_list]
    clean_data = [i for i in clean_data if i.isalpha()]
    clean_data = [i for i in clean_data if len(i)>1]
    return clean_data

def lemmatization(text_list):
    final_list = []
    lemmatizer = WordNetLemmatizer()
    for i in text_list:
        w = lemmatizer.lemmatize(i)
        final_list.append(w)
    return " ".join(final_list)

def clean_input_text(input_text):
    # Remove newline and tab characters
    cleaned_text = remove_blank(input_text)

    # Expand text (assuming fix function is defined)
    cleaned_text = expanding_text(cleaned_text)

    # Handle accented characters
    cleaned_text = handle_accented_chr(cleaned_text)

    # Tokenize and clean text
    tokens = word_tokenize(cleaned_text.lower())
    clean_data = [i for i in tokens if i not in punctuation]
    clean_data = [i for i in clean_data if i not in stopword_list]
    clean_data = [i for i in clean_data if i.isalpha()]
    clean_data = [i for i in clean_data if len(i) > 1]

    # Lemmatization
    cleaned_text = lemmatization(clean_data)

    return cleaned_text




# Load the saved tokenizer
with open("tokenizer.pkl", "rb") as tokenizer_file:
    tokenizer = pickle.load(tokenizer_file)

# Load the saved model
loaded_model = load_model("SmartLabel.keras")


def preprocess_input(input_text, tokenizer, max_sequence_length=400):
    # Clean the input text
    cleaned_text = clean_input_text(input_text)  # Assuming clean_input_text function is defined

    # Tokenize the cleaned text
    input_seq = tokenizer.texts_to_sequences([cleaned_text])

    # Pad the sequence
    padded_seq = pad_sequences(input_seq, padding='post', truncating='post', maxlen=max_sequence_length)

    return padded_seq

# @csrf_exempt
# def classify_text(request):
#     if request.method == 'POST':
#         print("Request Body:", request.body)
#         # Get the input text from the request body
#         input_text = request.POST.get('text')
#         print("Received Input Text:", input_text)
#
#         # Preprocess the text
#         cleaned_text = clean_input_text(input_text)
#         print("Cleaned Text:", cleaned_text)
#
#         # Tokenize and pad the sequence
#         preprocessed_input = preprocess_input(cleaned_text, tokenizer)
#         print("Preprocessed Input:", preprocessed_input)
#
#         # Predict the label
#         predictions = loaded_model.predict(preprocessed_input)
#         print("Predictions:", predictions)
#
#         result = np.argmax(predictions)
#         labels = ['paper','patent', 'meetings','seminar']
#
#         # Return the predicted label as JSON response
#         return JsonResponse({'label': labels[result]})
#     else:
#         return JsonResponse({'error': 'Only POST requests are allowed'})

import json

@csrf_exempt
def classify_text(request):
    if request.method == 'POST':
        # Retrieve the entire request body
        request_data = json.loads(request.body.decode('utf-8'))

        # Get the input text from the JSON data
        input_text = request_data.get('text')
        print("Received Input Text:", input_text)

        # Preprocess the text
        cleaned_text = clean_input_text(input_text)
        print("Cleaned Text:", cleaned_text)

        # Tokenize and pad the sequence
        preprocessed_input = preprocess_input(cleaned_text, tokenizer)
        print("Preprocessed Input:", preprocessed_input)

        # Predict the label
        predictions = loaded_model.predict(preprocessed_input)
        print("Predictions:", predictions)

        result = np.argmax(predictions)
        labels = ['paper', 'patent', 'meetings', 'seminar']

        # Return the predicted label as JSON response
        return JsonResponse({'label': labels[result]})
    else:
        return JsonResponse({'error': 'Only POST requests are allowed'})



def home_view(request):
    # You can customize the response here
    context = {'message': 'Welcome to your Smart Labeling API!'}
    return render(request, 'home.html', context)
