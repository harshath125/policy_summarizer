import os
import re
from flask import Flask, render_template, request, jsonify
import fitz  # PyMuPDF
from transformers import pipeline
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
import torch

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

# --- Model Loading ---
print("Loading summarization model... This might take a few minutes on the first run.")
try:
    device = 0 if torch.cuda.is_available() else -1
    # FIX: Use a smaller, distilled model to fit within free hosting memory limits
    abstractive_summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-6-6", device=device)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Could not load model with specific device, falling back. Error: {e}")
    # Fallback with the smaller model if the above fails
    abstractive_summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-6-6")
    print("Model loaded successfully on fallback.")


def preprocess_text(text):
    """Cleans the text by removing extra spaces and special characters."""
    text = text.replace('\n', ' ').replace('\r', '')
    text = re.sub(' +', ' ', text)
    return text.strip()

def summarize_extractively(text, sentence_count=7):
    """Generates an extractive summary using Sumy (LSA)."""
    if not text or len(text.split()) < 20:
        return ["Not enough text to generate key sentences."]
        
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = LsaSummarizer()
    summary = summarizer(parser.document, sentence_count)
    return [str(sentence) for sentence in summary]

def summarize_abstractive(text, detail_level=3):
    """Generates an abstractive summary using Hugging Face Transformers."""
    word_count = len(text.split())
    
    if not text or word_count < 40:
        return "Not enough text to generate a simplified summary for this section."

    # Dynamically adjust summary length based on detail level and text length
    # Levels: 1=Concise, 3=Balanced, 5=Exhaustive
    base_max = 80 + (detail_level * 20)
    base_min = 20 + (detail_level * 10)

    adjusted_max_len = min(base_max, int(word_count * 0.8))
    adjusted_min_len = min(base_min, int(word_count * 0.3))

    if adjusted_min_len >= adjusted_max_len:
        adjusted_min_len = int(adjusted_max_len * 0.5)

    try:
        summary_list = abstractive_summarizer(
            text, 
            max_length=adjusted_max_len, 
            min_length=adjusted_min_len, 
            do_sample=False,
            truncation=True
        )
        if summary_list:
            return summary_list[0]['summary_text']
        else:
            return "Could not generate a simplified summary for this section."
    except Exception as e:
        print(f"CRITICAL Error during abstractive summarization: {e}")
        return "An error occurred while generating the simplified summary."

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/summarize', methods=['POST'])
def summarize():
    if 'pdf' not in request.files:
        return render_template('index.html', error="No file part selected.")
        
    pdf_file = request.files['pdf']
    if pdf_file.filename == '':
        return render_template('index.html', error="No file selected for upload.")

    if pdf_file:
        try:
            # Extract text from PDF
            text = ""
            doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
            for page in doc:
                text += page.get_text()
            
            cleaned_text = preprocess_text(text)
            words = cleaned_text.split()
            total_words = len(words)

            num_chunks = int(request.form.get('chunks', 5))
            detail_level = int(request.form.get('detail', 3))
            
            chunk_size = total_words // num_chunks
            if chunk_size < 50: # Ensure chunks are not too small
                num_chunks = max(1, total_words // 50)
                chunk_size = total_words // num_chunks

            summaries = []
            for i in range(num_chunks):
                start_index = i * chunk_size
                end_index = (i + 1) * chunk_size
                chunk_text = " ".join(words[start_index:end_index])
                
                if chunk_text.strip():
                    extractive_summary = summarize_extractively(chunk_text, sentence_count=5)
                    abstractive_summary = summarize_abstractive(chunk_text, detail_level)
                    summaries.append({
                        'chunk_num': i + 1,
                        'extractive': extractive_summary,
                        'abstractive': abstractive_summary
                    })
            
            return render_template('result.html', summaries=summaries)

        except Exception as e:
            print(f"An error occurred: {e}")
            return render_template('index.html', error=f"An error occurred while processing the PDF: {e}")

    return render_template('index.html')

if __name__ == '__main__':
    # Make sure the 'uploads' directory exists
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)

