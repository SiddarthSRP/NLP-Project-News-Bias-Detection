# app.py
from flask import Flask, request, jsonify, render_template
from transformers import BertTokenizer, BertForSequenceClassification, pipeline
import torch
import requests
from bs4 import BeautifulSoup
from googlesearch import search
import os

app = Flask(__name__, template_folder=os.path.join(os.path.dirname(os.path.dirname(__file__)), 'frontend', 'templates'))

# Load the models and tokenizer
model_path = r"D:\NLP Project\backend\new_model"
model = BertForSequenceClassification.from_pretrained(model_path)
tokenizer = BertTokenizer.from_pretrained(model_path)
# Initialize the summarization pipeline
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
model.eval()

# Label map
label_map = {
    0: "left-leaning",
    1: "centrist/neutral",
    2: "right-leaning"
}

# Keywords for Indian news sites
INDIAN_NEWS_SITES = [
    "ndtv.com",                # Generally center-left
    "thehindu.com",            # Center-left
    "indiatoday.in",           # Centrist
    "timesofindia.indiatimes.com",  # Centrist
    "hindustantimes.com",      # Center-right
    "republicworld.com",       # Right-leaning
    "scroll.in",               # Left-leaning
    "news18.com",              # Center-right
    "theprint.in",             # Centrist
    "opindia.com",             # Right-leaning
    "thewire.in",              # Left-leaning
    "newslaundry.com",         # Left-leaning
    "deccanherald.com",        # Centrist
    "swarajyamag.com",         # Right-leaning
    "livemint.com"             # Centrist/economics focused
]

def generate_summary(text):
    try:
        # Truncate text if it's too long (BART model has a limit)
        max_length = 1024
        if len(text.split()) > max_length:
            text = ' '.join(text.split()[:max_length])
        
        # Generate summary
        summary = summarizer(text, max_length=150, min_length=50, do_sample=False)[0]['summary_text']
        return summary
    except Exception as e:
        print(f"Summarization error: {str(e)}")
        return None

# Function to scrape article content and generate summary
def scrape_article_text(url):
    try:
        response = requests.get(url, timeout=5)
        soup = BeautifulSoup(response.text, "html.parser")
        
        # Get the article title
        title = soup.find('title')
        title = title.get_text() if title else "Untitled Article"
        
        # Get the main content based on news source
        content = ""
        
        if "timesofindia.indiatimes.com" in url:
            content_div = soup.find('div', class_='M1rHh vkpDP')
            if content_div:
                content = content_div.get_text()
        elif "thehindu.com" in url:
            # The Hindu specific content extraction
            article_div = soup.find('div', {'id': 'content-body'}) or \
                         soup.find('div', class_='article') or \
                         soup.find('div', class_='content')
            if article_div:
                paragraphs = article_div.find_all('p')
                content = ' '.join([p.get_text() for p in paragraphs])
        else:
            paragraphs = soup.find_all("p")
            content = " ".join([p.get_text() for p in paragraphs])
        
        # Special handling for The Wire articles
        if "thewire.in" in url:
            words = content.split()
            if len(words) > 45:
                content = " ".join(words[45:])
        
        # Generate AI-powered summary
        summary = generate_summary(content) if content else "Unable to generate summary."
        if not summary:
            summary = "Unable to generate summary."
            
        return content.strip(), summary, title
    except Exception as e:
        print(f"Scraping error: {str(e)}")
        return "", "Unable to access article content.", "Untitled Article"

# Function to get news articles by keyword and classify bias
def get_articles_with_bias(keyword):
    articles = []
    for url in search(keyword + " site:" + " OR site:".join(INDIAN_NEWS_SITES), num_results=15):
        if any(site in url for site in INDIAN_NEWS_SITES):
            content, summary, title = scrape_article_text(url)
            if content:
                inputs = tokenizer(content, return_tensors="pt", truncation=True, max_length=512)
                with torch.no_grad():
                    logits = model(**inputs).logits
                    predicted_class = torch.argmax(logits, dim=1).item()
                    bias = label_map[predicted_class]
                    articles.append({
                        "url": url,
                        "bias": bias,
                        "summary": summary,
                        "title": title
                    })
    return articles

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/search", methods=["POST"])
def search_articles():
    keyword = request.form.get("keyword")
    filter_bias = request.form.get("filter")

    articles = get_articles_with_bias(keyword)

    if filter_bias and filter_bias != "all":
        articles = [a for a in articles if a["bias"] == filter_bias]

    return jsonify(articles)

if __name__ == "__main__":
    app.run(debug=True)
