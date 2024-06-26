from django.shortcuts import render
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('vader_lexicon')

sia = SentimentIntensityAnalyzer()

def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in filtered_tokens]
    return lemmatized_tokens

def analyze_sentiment(text):
    preprocessed_text = preprocess_text(text)
    sentiment_score = sia.polarity_scores(' '.join(preprocessed_text))
    if sentiment_score['compound'] >= 0.05:
        sentiment_label = 'Positive'
    elif sentiment_score['compound'] <= -0.05:
        sentiment_label = 'Negative'
    else:
        sentiment_label = 'Neutral'
    return sentiment_label

def index(request):
    if request.method == 'POST':
        text = request.POST.get('text', '')
        sentiment = analyze_sentiment(text)
        return render(request, 'analyzer/index.html', {'sentiment': sentiment, 'text': text})
    return render(request, 'analyzer/index.html')
