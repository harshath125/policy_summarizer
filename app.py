from flask import Flask, render_template, request
import fitz  # PyMuPDF
from transformers import pipeline
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
import os
import torch
import math
import re

# Initialize the Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

# --- Model Loading ---
print("Loading summarization model... This might take a few minutes on the first run.")
try:
    device = 0 if torch.cuda.is_available() else -1
    abstractive_summarizer = pipeline("summarization", model="t5-small", device=device)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Could not load model with specific device, falling back. Error: {e}")
    abstractive_summarizer = pipeline("summarization", model="t5-small")
    print("Model loaded successfully on fallback.")


def preprocess_text(text):
    """Cleans text by removing extra whitespace and non-standard characters."""
    text = text.replace('\n', ' ')  # Replace newlines with spaces
    text = re.sub(' +', ' ', text)  # Replace multiple spaces with a single space
    return text.strip()

def extract_text_from_pdf(file_stream):
    """Extracts and preprocesses text from a PDF file stream."""
    text = ""
    try:
        with fitz.open(stream=file_stream.read(), filetype="pdf") as doc:
            for page in doc:
                text += page.get_text()
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return None
    return preprocess_text(text)

def summarize_extractively(text, sentence_count=5):
    """Generates an extractive summary and returns a list of sentences."""
    if not text or len(text.split()) < 20:
        return ["Not enough text to generate a quick overview."]
    try:
        parser = PlaintextParser.from_string(text, Tokenizer("english"))
        summarizer = LsaSummarizer()
        summary = summarizer(parser.document, sentence_count)
        return [str(sentence) for sentence in summary]
    except Exception as e:
        print(f"Error in extractive summarization: {e}")
        return ["Could not generate extractive summary for this chunk."]

def summarize_abstractive(text, max_len=150, min_len=40):
    """Generates an abstractive summary with robust checks."""
    words = text.split()
    word_count = len(words)

    if word_count < 50: # Stricter check for T5 model
        return "Not enough text for a high-quality summary. This section should have at least 50 words."

    if max_len > word_count:
        max_len = word_count
    if min_len >= max_len:
        min_len = max_len // 2
        
    if min_len < 10: # Absolute minimum for the model
        min_len = 10

    try:
        summary = abstractive_summarizer(
            text,
            max_length=max_len,
            min_length=min_len,
            do_sample=False,
            truncation=True
        )
        if summary and 'summary_text' in summary[0]:
            return summary[0]['summary_text']
        else:
            return "Could not generate a summary for this section (model returned an unexpected result)."
    except Exception as e:
        print(f"CRITICAL Error during abstractive summarization: {e}")
        return f"A model error occurred. The input for this section might be too complex or short. [Error: {e}]"


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/summarize', methods=['POST'])
def summarize():
    if 'pdf' not in request.files:
        return render_template('index.html', error="No file part in the request.")

    pdf_file = request.files['pdf']
    num_chunks = int(request.form.get('chunks', 5))
    detail_level = int(request.form.get('detail', 3))

    # --- Map detail level to summarization parameters ---
    detail_map = {
        1: {'min': 20, 'max': 60, 'text': 'Concise'},
        2: {'min': 30, 'max': 90, 'text': 'Brief'},
        3: {'min': 50, 'max': 150, 'text': 'Balanced'},
        4: {'min': 80, 'max': 200, 'text': 'Detailed'},
        5: {'min': 120, 'max': 300, 'text': 'Exhaustive'}
    }
    summary_params = detail_map.get(detail_level)
    min_len = summary_params['min']
    max_len = summary_params['max']
    detail_text = summary_params['text']

    if pdf_file.filename == '':
        return render_template('index.html', error="No file selected for uploading.")

    if pdf_file and pdf_file.filename.lower().endswith('.pdf'):
        try:
            text = extract_text_from_pdf(pdf_file.stream)
            if not text or not text.strip():
                return render_template('result.html', error="Could not extract text from the PDF.")

            words = text.split()
            chunk_size = math.ceil(len(words) / num_chunks)
            if chunk_size == 0:
                 return render_template('result.html', error="Document is too short to be divided into chunks.")

            summaries = []
            for i in range(num_chunks):
                chunk_text = " ".join(words[i*chunk_size : (i+1)*chunk_size])
                if not chunk_text.strip(): continue

                extractive_sum = summarize_extractively(chunk_text)
                abstractive_sum = summarize_abstractive(chunk_text, max_len=max_len, min_len=min_len)

                summaries.append({
                    'chunk_num': i + 1,
                    'extractive': extractive_sum,
                    'abstractive': abstractive_sum
                })

            return render_template('result.html', summaries=summaries, total_chunks=num_chunks, detail_level_text=detail_text)

        except Exception as e:
            print(f"An error occurred during summarization route: {e}")
            return render_template('result.html', error="An unexpected error occurred.")
    else:
        return render_template('index.html', error="Invalid file type. Please upload a PDF.")

if __name__ == '__main__':
    os.makedirs('uploads', exist_ok=True)
    app.run(debug=True)

