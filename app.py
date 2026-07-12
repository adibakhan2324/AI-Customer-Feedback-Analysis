from flask import Flask, render_template, request
import torch
import torch.nn.functional as F
import pickle
import time
import re
import json
import os

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from transformers import BertTokenizer, BertForSequenceClassification

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

from language_rules import (
    POSITIVE_SLANG,
    NEGATIVE_SLANG,
    PROFANITY,
    POSITIVE_EMOJIS,
    NEGATIVE_EMOJIS,
    NEGATION,
    CONTRAST,
    SARCASM,
    POSITIVE_PHRASES,
    NEGATIVE_PHRASES,
    MIXED_PHRASES,
    POSITIVE_WORDS,
    NEGATIVE_WORDS,
    NEUTRAL_SLANGS,
    MIXED_WORDS
    
)

from business_recommendation import generate_recommendation

from customer_issue_detection import detect_customer_issues
from business_report import generate_business_report

# ==========================================
# Statistics Storage Functions
# ==========================================

def load_stats():

    if os.path.exists("stats.json"):

        with open("stats.json","r") as file:
            return json.load(file)

    return {
        "total_reviews":0,
        "positive_reviews":0,
        "negative_reviews":0
    }


def save_stats(stats):

    with open("stats.json","w") as file:
        json.dump(stats,file)
        
# ==========================================
# Review History
# ==========================================

def load_history():

    if os.path.exists("history.json"):

        try:

            with open("history.json","r") as file:
                return json.load(file)

        except json.JSONDecodeError:

            return []

    return []


def save_history(history):

    with open("history.json","w") as file:
        json.dump(history, file, indent=4)

# ==========================================
# Flask App
# ==========================================

app = Flask(__name__)

# ==========================================
# Load BERT Model
# ==========================================

tokenizer = BertTokenizer.from_pretrained("models/bert_model")

bert_model = BertForSequenceClassification.from_pretrained(
    "models/bert_model"
)

bert_model.eval()

print("✅ BERT model loaded successfully!")

# ==========================================
# Load LSTM Model
# ==========================================

lstm_model = load_model("models/lstm_model.keras")

with open("models/lstm_tokenizer.pkl", "rb") as file:
    lstm_tokenizer = pickle.load(file)

print("✅ LSTM model loaded successfully!")

# ==========================================
# NLP Preprocessing
# ==========================================

stop_words = set(stopwords.words("english"))

lemmatizer = WordNetLemmatizer()

# ==========================================
# Clean Review
# ==========================================

def clean_text(text):

    text = text.lower()

    positive_emojis = {
        "😍": " love ",
        "😊": " happy ",
        "😁": " happy ",
        "😄": " happy ",
        "❤️": " love ",
        "❤": " love ",
        "👍": " good "
    }

    negative_emojis = {
        "😞": " sad ",
        "😢": " sad ",
        "😭": " sad ",
        "😡": " angry ",
        "👎": " bad "
    }

    for emoji, word in positive_emojis.items():
        text = text.replace(emoji, word)

    for emoji, word in negative_emojis.items():
        text = text.replace(emoji, word)

    text = re.sub(r'[^a-zA-Z\s]', ' ', text)

    text = re.sub(r'\s+', ' ', text)

    words = text.split()

    words = [
        word
        for word in words
        if word not in stop_words
    ]

    words = [
        lemmatizer.lemmatize(word)
        for word in words
    ]

    return " ".join(words).strip()


# ==========================================
# Advanced Language Analysis
# ==========================================

def analyze_language(review):

    text = review.lower()

    findings = []


    # Positive words
    positive_words = [

        "good",
        "great",
        "excellent",
        "amazing",
        "awesome",
        "beautiful",
        "perfect",
        "love",
        "wonderful",
        "best",
        "satisfied",
        "recommend",
        "premium",
        "fast",
        "easy",
        "helpful",
        "fantastic"

    ]


    # Negative words
    negative_words = [

        "bad",
        "worst",
        "terrible",
        "poor",
        "late",
        "delay",
        "broken",
        "damaged",
        "problem",
        "issue",
        "hate",
        "disappointed",
        "slow",
        "refund",
        "complaint",
        "waste"

    ]
    
        # Strong negative phrases
    strong_negative_phrases = [

        "broke my product",
        "broken on arrival",
        "damaged on arrival",
        "quality became terrible",
        "quality became bad",
        "performance is poor",
        "stopped working",
        "not working",
        "does not work",
        "worst experience",
        "terrible service",
        "poor quality"

    ]


    # Detect strong negative situations
    for phrase in strong_negative_phrases:

        if phrase in text:

            findings.append(
                f"🚨 Strong negative situation detected: {phrase}"
            )
        # Mixed sentiment detection

    positive_found = False
    negative_found = False


    for word in positive_words:
        if word in text:
            positive_found = True


    for word in negative_words:
        if word in text:
            negative_found = True


    if positive_found and negative_found:

        findings.append(
            "⚖ Mixed sentiment detected: Positive and negative opinions found"
        )        

    # Detect positive words
    for word in positive_words:

        if word in text:

            findings.append(
                f"😊 Positive word detected: {word}"
            )


    # Detect negative words
    for word in negative_words:

        if word in text:

            findings.append(
                f"😞 Negative word detected: {word}"
            )


    # Detect contrast
    contrast_words = [

        "but",
        "however",
        "although",
        "though",
        "yet"

    ]


    for word in contrast_words:

        if word in text:

            findings.append(
                f"⚖ Mixed opinion indicator detected: {word}"
            )


    # Detect emojis

    for emoji in POSITIVE_EMOJIS:

        if emoji in review:

            findings.append(
                f"😊 Positive emoji detected: {emoji}"
            )


    for emoji in NEGATIVE_EMOJIS:

        if emoji in review:

            findings.append(
                f"😞 Negative emoji detected: {emoji}"
            )


    # If nothing detected

    if len(findings) == 0:

        findings.append(
            "ℹ No special language pattern detected"
        )


    return findings

# ==========================================
# Detect Complex / Mixed Reviews
# ==========================================

def detect_complex_review(review):

    review = review.lower()


    positive_words = [

        "good",
        "great",
        "excellent",
        "amazing",
        "awesome",
        "love",
        "loved",
        "perfect",
        "fantastic",
        "best",
        "happy",
        "wonderful",
        "beautiful",
        "nice",
        "brilliant",
        "outstanding",
        "impressive",
        "premium",
        "reliable",
        "comfortable",
        "satisfied",
        "recommend",
        "recommended",
        "enjoy",
        "enjoyed",
        "success",
        "successful",
        "helpful",
        "friendly",
        "fast",
        "easy",
        "worth",
        "valuable"

    ]


    negative_words = [

        "bad",
        "worst",
        "broken",
        "terrible",
        "awful",
        "poor",
        "late",
        "delay",
        "delayed",
        "damaged",
        "problem",
        "issue",
        "hate",
        "shit",
        "horrible",
        "disappointing",
        "disappointed",
        "frustrating",
        "frustrated",
        "angry",
        "slow",
        "failure",
        "failed",
        "useless",
        "waste",
        "fake",
        "scam",
        "refund",
        "cancel",
        "complaint",
        "unacceptable",
        "pathetic"

    ]


    positive = any(
        word in review 
        for word in positive_words
    )


    negative = any(
        word in review 
        for word in negative_words
    )


    # Detect contrast words

    contrast_words = [

        "but",
        "however",
        "although",
        "though",
        "yet",
        "except"

    ]


    contrast = any(
        word in review
        for word in contrast_words
    )


    # Mixed sentiment condition

    if positive and negative:

        return True


    if contrast and (positive or negative):

        return True


    return False


# ==========================================
# Business Recommendation Engine
# ==========================================

def business_recommendation(sentiment):

    if "Positive" in sentiment:
        return (
            "💼 Business Recommendation"
            "<br><br>"
            "Customers are generally satisfied."
            "<br>"
            "Continue maintaining product quality and service."
        )

    elif "Negative" in sentiment:
        return (
            "💼 Business Recommendation"
            "<br><br>"
            "Customer dissatisfaction detected."
            "<br>"
            "Review product quality, delivery, or customer support."
        )

    else:
        return (
            "💼 Business Recommendation"
            "<br><br>"
            "Customer opinion is neutral."
            "<br>"
            "Collect additional feedback before taking action."
        )


# ==========================================
# Confidence Level
# ==========================================

def confidence_level(confidence):

    if confidence >= 90:
        return "High 🟢"

    elif confidence >= 75:
        return "Medium 🟡"

    return "Low 🔴"

# ==========================================
# AI Prediction Correction Layer
# ==========================================

def correct_prediction(review, sentiment):
    
    print("Correction layer running:", review)

    text = review.lower()
    
    # FIRST: Strong negative situations

    if "quality became terrible" in text:
        return "😞 Negative Review"

    if "broke my product" in text:
        return "😞 Negative Review"


    # Strong negative phrases (highest priority)

    negative_phrases = [

    "very bad",
    "really bad",
    "extremely bad",
    "too bad",
    "so bad",

    "poor quality",
    "bad quality",
    "worst quality",
    "terrible quality",
    "low quality",
    "cheap quality",

    "poor service",
    "bad service",
    "worst service",
    "terrible service",
    "slow service",
    "late delivery",

    "worst product",
    "terrible product",
    "horrible product",
    "damaged product",
    "broken product",

    "not satisfied",
    "not happy",
    "not worth",
    "not recommended",
    "never buy",
    "never purchase",

    "waste of money",
    "complete waste",
    "money wasted",

    "very disappointed",
    "extremely disappointed",
    "disappointing experience",

    "hate this",
    "hate it",
    "hate product",

    "customer complaint",
    "bad experience",
    "worst experience",

    "does not work",
    "not working",
    "stopped working",
    "doesn't work"

]
    
    strong_negative_phrases = [

    "stopped working",
    "not working",
    "does not work",
    "doesn't work",
    "failed to work",

    "arrived damaged",
    "arrived broken",
    "received damaged",
    "received broken",

    "broken on arrival",
    "damaged on arrival"


]
    


    # Strong positive phrases

    positive_phrases = [

    "very good",
    "really good",
    "extremely good",

    "good quality",
    "excellent quality",
    "high quality",
    "premium quality",
    "best quality",

    "excellent service",
    "great service",
    "amazing service",
    "fast delivery",

    "amazing product",
    "excellent product",
    "great product",
    "wonderful product",
    "perfect product",

    "love this",
    "love it",
    "really love",

    "highly recommend",
    "strongly recommend",

    "worth buying",
    "worth the money",
    "value for money",

    "very satisfied",
    "extremely satisfied",

    "best experience",
    "amazing experience",

    "works perfectly",
    "works great",

    "customer friendly",
    "excellent support",
    
    "working perfectly",
    "works perfectly",
    "works great",
    "working well",
    "works fine"

]


    for phrase in negative_phrases:

        if phrase in text:
            return "😞 Negative Review"


    for phrase in positive_phrases:

        if phrase in text:
            return "😊 Positive Review"



    # Negative words

    negative_words = [

"bad",
"worst",
"terrible",
"horrible",
"awful",
"poor",
"hate",
"dislike",
"angry",
"broken",
"damage",
"damaged",
"useless",
"waste",
"fake",
"fraud",
"slow",
"late",
"delay",
"problem",
"issue",
"failure",
"failed",
"refund",
"cancel",
"complaint",
"disappointed",
"disappointing",
"annoying",
"frustrating",
"frustrated",
"cheap",
"worst",
"ugly",
"unhappy",
"wrong"

]


    # Positive words

    positive_words = [

"good",
"great",
"excellent",
"amazing",
"awesome",
"wonderful",
"perfect",
"fantastic",
"love",
"liked",
"like",
"best",
"beautiful",
"nice",
"happy",
"satisfied",
"recommend",
"success",
"brilliant",
"impressive",
"comfortable",
"reliable",
"premium",
"helpful",
"friendly",
"fast",
"easy",
"clean",
"enjoy",
"enjoyed",
"favorite",
"super",
"outstanding"

]


    negative_score = 0
    positive_score = 0


    for word in negative_words:

        if word in text:
            negative_score += 1


    for word in positive_words:

        if word in text:
            positive_score += 1



    # Negation handling

    negations = [
        "not",
        "never",
        "no",
        "without"
    ]


    for neg in negations:

        if neg in text:

            negative_score += 1



    # Emoji analysis

    negative_emojis = [

    "😡",
    "😠",
    "😞",
    "😔",
    "😢",
    "😭",
    "😤",
    "🤬",
    "👎",
    "💔",
    "😒",
    "🙄",
    "😕",
    "😟",
    "😰",
    "😱",
    "🤦",
    "🤦‍♂️",
    "🤦‍♀️",
    "☹️",
    "😣",
    "😖",
    "❌",
    "✖️",
    "🚫",

]


    positive_emojis = [

    "😊",
    "😀",
    "😁",
    "😃",
    "😄",
    "😍",
    "🥰",
    "❤️",
    "❤",
    "💕",
    "💖",
    "👍",
    "👏",
    "🎉",
    "🔥",
    "⭐",
    "🌟",
    "✨",
    "🤩",
    "😎",
    "👌",
    "💯",
    "✅",
    "✔️",
    "☑️"

]


    for emoji in negative_emojis:

        if emoji in review:
            negative_score += 1


    for emoji in positive_emojis:

        if emoji in review:
            positive_score += 1
            
        # ==========================================
    # Contrast Based Mixed Sentiment Handling
    # ==========================================

    contrast_words = [
        "but",
        "however",
        "although",
        "though",
        "yet"
    ]


    negative_after_contrast = [

        "bad",
        "worst",
        "terrible",
        "poor",
        "late",
        "delay",
        "broken",
        "disappointed",
        "slow",
        "problem",
        "issue",
        "hate",
        "awful",
        "worse"

    ]


    for contrast in contrast_words:

        if contrast in text:
            
            print("Contrast found:", contrast)

            after_contrast = text.split(contrast, 1)[1]


            for word in negative_after_contrast:

                if word in after_contrast:

                    return "😞 Negative Review"



    # ==========================================
    # Final decision
    # ==========================================

        if negative_score > positive_score:

            return "😞 Negative Review"


        elif positive_score > negative_score:

            return "😊 Positive Review"


        return sentiment


# ==========================================
# LSTM Prediction
# ==========================================

def predict_lstm(review):

    start_time = time.time()

    cleaned_review = clean_text(review)

    sequence = lstm_tokenizer.texts_to_sequences([cleaned_review])

    padded = pad_sequences(
        sequence,
        maxlen=150,
        padding="post",
        truncating="post"
    )

    prediction = lstm_model.predict(
        padded,
        verbose=0
    )

    score = float(prediction[0][0])

    if score >= 0.65:
        sentiment = "😊 Positive Review"
        confidence = score * 100

    elif score <= 0.35:
        sentiment = "😞 Negative Review"
        confidence = (1 - score) * 100

    else:
        sentiment = "😐 Neutral Review"
        confidence = max(score, 1 - score) * 100

    level = confidence_level(confidence)

    warning = ""

    if detect_complex_review(review):
        warning = (
            "<br><br>"
            "⚠ Mixed or ambiguous language detected."
            "<br>"
            "Prediction should be interpreted carefully."
        )
        
    sentiment = correct_prediction(review, sentiment)

    recommendation = business_recommendation(sentiment)

    # ==========================
    # AI Language Analysis
    # ==========================

    analysis = analyze_language(review)


    analysis_text = (
        "<br><br><b>🧠 AI Language Analysis</b><br>"
    )


    for item in analysis:

        analysis_text += item + "<br>"

    prediction_time = round(time.time() - start_time, 3)
    
    speed = "Fast"

    prediction_text = (
    f"{sentiment}"
    f"<br><br>Confidence : {confidence:.2f}%"
    f"<br><br>Confidence Level : {level}"
    f"<br><br>⚡ Prediction Time : {prediction_time} sec"
    f"<br><br>{recommendation}"
    f"{warning}"
    f"{analysis_text}"
    f"<br><br>🤖 Model Used : Bi-LSTM"
)

    return prediction_text, confidence, prediction_time
    
    # ==========================================
# BERT Prediction
# ==========================================

def predict_bert(review):

    start_time = time.time()

    inputs = tokenizer(
        review,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    with torch.no_grad():
        outputs = bert_model(**inputs)

    prediction_id = torch.argmax(outputs.logits, dim=1).item()
    
    print("BERT prediction id:", prediction_id)

    probabilities = F.softmax(outputs.logits, dim=1)

    confidence = torch.max(probabilities).item() * 100

    level = confidence_level(confidence)

    if prediction_id == 1:
        sentiment = "😊 Positive Review"
    else:
        sentiment = "😞 Negative Review"

    sentiment = correct_prediction(review, sentiment)

    warning = ""

    if detect_complex_review(review):
        warning = (
            "<br><br>"
            "⚠ Mixed or ambiguous language detected."
            "<br>"
            "Prediction should be interpreted carefully."
        )

    recommendation = business_recommendation(sentiment)

    analysis = analyze_language(review)
    
    analysis_text = ""

    if analysis:

        analysis_text = (
            "<br><br><b>🧠 AI Language Analysis</b><br>"
        )

        for item in analysis:
            analysis_text += item + "<br>"

    prediction_time = round(time.time() - start_time, 3)
    
    speed = "Fast"

    return (
    f"{sentiment}"
    f"<br><br>Confidence : {confidence:.2f}%"
    f"<br><br>Confidence Level : {level}"
    f"<br><br>⚡ Speed : {speed}"
    f"<br><br>⚡ Prediction Time : {prediction_time} sec"
    f"<br><br>{recommendation}"
    f"{warning}"
    f"{analysis_text}"
    f"<br><br>🤖 Model Used : BERT",
    confidence,
    prediction_time
)
    
    # ==========================================
# Home Page
# ==========================================

@app.route("/", methods=["GET", "POST"])
def home():

    prediction = ""
    review = ""
    
    recommendation_text = ""
    
    confidence = 0
    speed = 0
    
    lstm_confidence = 0
    lstm_speed = 0
    
    analysis_text = ""
    
    recommendations = []
    
    recommendations = ""
    
    customer_issues = []
    
    business_report = ""
    
    best_model = ""
    comparison_reason = ""
    
    total_reviews = 0
    positive_reviews = 0
    negative_reviews = 0
    
    bert_prediction = ""

    lstm_prediction = ""

    history = []

    if request.method == "POST":

            review = request.form["review"]
            
            # Customer Issue Detection
            customer_issues = detect_customer_issues(review)
            
            # AI Business Report Generation

            sentiment_result = bert_prediction.split("<br>")[0].strip()

            business_report = generate_business_report(
                sentiment_result,
                customer_issues
     )
            
            stats = load_stats()

            stats["total_reviews"] += 1
            

            bert_prediction, confidence, speed = predict_bert(review)

            if "Positive" in bert_prediction:

                stats["positive_reviews"] += 1


            elif "Negative" in bert_prediction:

                stats["negative_reviews"] += 1


            lstm_prediction, lstm_confidence, lstm_speed = predict_lstm(review)
            
            analysis = analyze_language(review)

            if analysis:

                analysis_text = "<br><br><b>🧠 AI Language Analysis</b><br>"

                for item in analysis:
                    analysis_text += item + "<br>"

            else:

                analysis_text = (
                    "<br><br>🧠 AI Language Analysis<br>"
                    "No special language patterns detected."
           )
                
            recommendations = generate_recommendation(review)

            recommendation_text = "<br><br><b>💼 Business Recommendation</b><br>"
 
            if recommendations:
                for rec in recommendations:
                    recommendation_text += "• " + rec + "<br>"
            else:
                recommendation_text += "No recommendation generated."

            
            # ==========================
            # Save Review History
            # ==========================

            history = load_history()


            history.append({

                "review": review,

                "sentiment": bert_prediction.split("<br>")[0].strip(),

                "confidence": round(confidence,2),

                "model": "BERT AI"

      })


            save_history(history)

            save_stats(stats)
            
            if confidence >= lstm_confidence:

                best_model = "🏆 Best Model: BERT AI"

                comparison_reason = (
                    "Higher confidence and faster prediction compared to Bi-LSTM."
                )

            else:

                best_model = "🏆 Best Model: Bi-LSTM AI"

                comparison_reason = (
                    "Bi-LSTM produced better confidence for this review."
                )

            prediction = f"""
<div class='model-card'>

<h2>🤖 BERT AI Model</h2>

{bert_prediction}

</div>

<div class='model-card'>

<h2>🧠 Bi-LSTM AI Model</h2>

{lstm_prediction}

</div>
"""
# ==========================
# Load statistics
# ==========================

    stats = load_stats()

    total_reviews = stats["total_reviews"]

    positive_reviews = stats["positive_reviews"]

    negative_reviews = stats["negative_reviews"]
    
    history = load_history()

    history = history[-5:]
    
    print("Recommendation variable:", recommendation_text)


    return render_template(
    "dashboard.html",
    
    prediction=prediction,
    review=review,

    # BERT Result
    bert_prediction=bert_prediction,
    bert_confidence=confidence,
    bert_speed=speed,

    # Bi-LSTM Result
    lstm_prediction=lstm_prediction,
    lstm_confidence=lstm_confidence,
    lstm_speed=lstm_speed,

    # Decision
    best_model=best_model,
    comparison_reason=comparison_reason,

    # Statistics
    total_reviews=total_reviews,
    positive_reviews=positive_reviews,
    negative_reviews=negative_reviews,

    # History
    history=history,

    # Analysis
    analysis_text=analysis_text,
    
    # Business Recommendation
    recommendation_text=recommendation_text,
    
    # Business Report
    customer_issues=customer_issues,
    business_report=business_report
)


# ==========================================
# Run Flask
# ==========================================

if __name__ == "__main__":
    app.run(debug=True)