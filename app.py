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


from transformers import (
    BertTokenizer,
    BertForSequenceClassification
)


from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


from language_rules import (
    POSITIVE_EMOJIS,
    NEGATIVE_EMOJIS
)


from business_recommendation import generate_recommendation

from customer_issue_detection import detect_customer_issues

from business_report import generate_business_report



# =====================================================
# CustomerPulse AI
# Flask Application
# =====================================================



app = Flask(__name__)




# =====================================================
# Statistics Management
# =====================================================


def load_stats():

    if os.path.exists("stats.json"):

        try:

            with open("stats.json", "r") as file:

                return json.load(file)

        except:

            pass


    return {

        "total_reviews": 0,

        "positive_reviews": 0,

        "negative_reviews": 0

    }



def save_stats(stats):

    with open("stats.json", "w") as file:

        json.dump(
            stats,
            file,
            indent=4
        )




# =====================================================
# Review History Management
# =====================================================


def load_history():

    if os.path.exists("history.json"):

        try:

            with open("history.json", "r") as file:

                return json.load(file)


        except:

            return []


    return []





def save_history(history):

    with open("history.json", "w") as file:

        json.dump(
            history,
            file,
            indent=4
        )




# =====================================================
# Load BERT Model
# =====================================================


HF_MODEL = "khanadiba263/customerpulse-bert-sentiment"

tokenizer = BertTokenizer.from_pretrained(HF_MODEL)

bert_model = BertForSequenceClassification.from_pretrained(HF_MODEL)

bert_model.eval()

print("✅ BERT model loaded successfully from Hugging Face")




# =====================================================
# Load Bi-LSTM Model
# =====================================================


lstm_model = load_model("models/lstm_model.keras")

with open(
    "models/lstm_tokenizer.pkl",
    "rb"
) as file:
    lstm_tokenizer = pickle.load(file)

print("✅ Bi-LSTM model loaded successfully")



# =====================================================
# NLP Tools
# =====================================================


stop_words = set(
    stopwords.words("english")
)



lemmatizer = WordNetLemmatizer()




# =====================================================
# Text Cleaning
# =====================================================


def clean_text(text):


    text = text.lower()



    emoji_map = {


        "😊": " happy ",

        "😍": " love ",

        "😁": " happy ",

        "😄": " happy ",

        "❤️": " love ",

        "👍": " good ",


        "😞": " sad ",

        "😢": " sad ",

        "😭": " sad ",

        "😡": " angry ",

        "👎": " bad "

    }



    for emoji, word in emoji_map.items():

        text = text.replace(
            emoji,
            word
        )



    text = re.sub(
        r"[^a-zA-Z\s]",
        " ",
        text
    )


    text = re.sub(
        r"\s+",
        " ",
        text
    )



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

# =====================================================
# Advanced Language Analysis
# =====================================================


def analyze_language(review):

    text = review.lower()

    findings = []



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



    strong_negative_phrases = [

        "broke my product",

        "broken on arrival",

        "damaged on arrival",

        "stopped working",

        "not working",

        "does not work",

        "worst experience",

        "terrible service",

        "poor quality"

    ]



    for phrase in strong_negative_phrases:

        if phrase in text:

            findings.append(
                f"🚨 Strong negative situation detected: {phrase}"
            )



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



    for word in positive_words:

        if word in text:

            findings.append(
                f"😊 Positive word detected: {word}"
            )



    for word in negative_words:

        if word in text:

            findings.append(
                f"😞 Negative word detected: {word}"
            )




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



    if len(findings) == 0:

        findings.append(
            "ℹ No special language pattern detected"
        )


    return findings






# =====================================================
# Detect Mixed / Complex Reviews
# =====================================================


def detect_complex_review(review):


    text = review.lower()



    positive_words = [

        "good",
        "great",
        "excellent",
        "amazing",
        "awesome",
        "love",
        "perfect",
        "fantastic",
        "best",
        "happy",
        "wonderful",
        "beautiful",
        "nice",
        "premium",
        "reliable",
        "satisfied",
        "recommend",
        "helpful",
        "fast",
        "easy"

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
        "damaged",
        "problem",
        "issue",
        "hate",
        "slow",
        "failure",
        "refund",
        "complaint",
        "unacceptable"

    ]



    positive = any(

        word in text

        for word in positive_words

    )



    negative = any(

        word in text

        for word in negative_words

    )



    contrast = any(

        word in text

        for word in [

            "but",
            "however",
            "although",
            "though",
            "yet"

        ]

    )



    if positive and negative:

        return True



    if contrast and (positive or negative):

        return True



    return False






# =====================================================
# Confidence Level
# =====================================================


def confidence_level(confidence):


    if confidence >= 90:

        return "High 🟢"



    elif confidence >= 75:

        return "Medium 🟡"



    else:

        return "Low 🔴"






# =====================================================
# AI Prediction Correction Layer
# =====================================================


def correct_prediction(review, sentiment):


    text = review.lower()



    negative_phrases = [

        "very bad",

        "really bad",

        "poor quality",

        "bad quality",

        "worst quality",

        "terrible service",

        "poor service",

        "late delivery",

        "broken product",

        "damaged product",

        "not satisfied",

        "not happy",

        "not worth",

        "never buy",

        "waste of money",

        "very disappointed",

        "hate this",

        "does not work",

        "not working"

    ]



    positive_phrases = [

        "very good",

        "excellent quality",

        "good quality",

        "high quality",

        "amazing product",

        "great product",

        "love this",

        "love it",

        "highly recommend",

        "worth buying",

        "very satisfied",

        "best experience",

        "works perfectly",

        "works great"

    ]



    for phrase in negative_phrases:


        if phrase in text:

            return "😞 Negative Review"




    for phrase in positive_phrases:


        if phrase in text:

            return "😊 Positive Review"



    return sentiment

# =====================================================
# Business Recommendation
# =====================================================


def business_recommendation(sentiment):


    if "Positive" in sentiment:


        return (

            "💼 Business Recommendation"
            "<br><br>"
            "Customers are satisfied."
            "<br>"
            "Continue maintaining product quality and service."

        )



    elif "Negative" in sentiment:


        return (

            "💼 Business Recommendation"
            "<br><br>"
            "Customer dissatisfaction detected."
            "<br>"
            "Improve product quality, delivery, and support."

        )



    else:


        return (

            "💼 Business Recommendation"
            "<br><br>"
            "Customer opinion is neutral."
            "<br>"
            "Collect more feedback."

        )






# =====================================================
# LSTM Prediction
# =====================================================


def predict_lstm(review):


    start_time = time.time()



    cleaned_review = clean_text(review)



    sequence = lstm_tokenizer.texts_to_sequences(
        [cleaned_review]
    )



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



    score = float(
        prediction[0][0]
    )



    if score >= 0.65:


        sentiment = "😊 Positive Review"

        confidence = score * 100



    elif score <= 0.35:


        sentiment = "😞 Negative Review"

        confidence = (1-score) * 100



    else:


        sentiment = "😐 Neutral Review"

        confidence = max(score,1-score) * 100




    sentiment = correct_prediction(
        review,
        sentiment
    )



    level = confidence_level(
        confidence
    )



    prediction_time = round(

        time.time()-start_time,

        3

    )



    warning = ""



    if detect_complex_review(review):


        warning = (

            "<br><br>"
            "⚠ Mixed or ambiguous language detected."
            "<br>"
            "Prediction should be interpreted carefully."

        )



    recommendation = business_recommendation(
        sentiment
    )



    analysis = analyze_language(
        review
    )



    analysis_text = (

        "<br><br>"
        "<b>🧠 AI Language Analysis</b>"
        "<br>"

    )



    for item in analysis:


        analysis_text += item + "<br>"




    result = (

        f"{sentiment}"

        f"<br><br>Confidence : {confidence:.2f}%"

        f"<br><br>Confidence Level : {level}"

        f"<br><br>⚡ Prediction Time : {prediction_time} sec"

        f"<br><br>{recommendation}"

        f"{warning}"

        f"{analysis_text}"

        f"<br><br>🤖 Model Used : Bi-LSTM"

    )



    return result, confidence, prediction_time







# =====================================================
# BERT Prediction
# =====================================================


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




    prediction_id = torch.argmax(

        outputs.logits,

        dim=1

    ).item()



    probabilities = F.softmax(

        outputs.logits,

        dim=1

    )



    confidence = torch.max(

        probabilities

    ).item()*100




    if prediction_id == 1:


        sentiment = "😊 Positive Review"



    else:


        sentiment = "😞 Negative Review"




    sentiment = correct_prediction(

        review,

        sentiment

    )



    level = confidence_level(
        confidence
    )



    prediction_time = round(

        time.time()-start_time,

        3

    )



    warning = ""



    if detect_complex_review(review):


        warning = (

            "<br><br>"
            "⚠ Mixed or ambiguous language detected."
            "<br>"
            "Prediction should be interpreted carefully."

        )




    recommendation = business_recommendation(
        sentiment
    )



    analysis = analyze_language(
        review
    )



    analysis_text = (

        "<br><br>"
        "<b>🧠 AI Language Analysis</b>"
        "<br>"

    )



    for item in analysis:


        analysis_text += item + "<br>"





    result = (

        f"{sentiment}"

        f"<br><br>Confidence : {confidence:.2f}%"

        f"<br><br>Confidence Level : {level}"

        f"<br><br>⚡ Prediction Time : {prediction_time} sec"

        f"<br><br>{recommendation}"

        f"{warning}"

        f"{analysis_text}"

        f"<br><br>🤖 Model Used : BERT"

    )



    return result, confidence, prediction_time

# =====================================================
# Home Route
# =====================================================


@app.route("/", methods=["GET", "POST"])
def home():


    prediction = ""

    review = ""


    bert_prediction = ""

    lstm_prediction = ""


    bert_confidence = 0

    lstm_confidence = 0


    bert_speed = 0

    lstm_speed = 0



    best_model = ""

    comparison_reason = ""



    recommendation_text = ""

    business_report = ""

    customer_issues = []



    total_reviews = 0

    positive_reviews = 0

    negative_reviews = []



    history = []




    if request.method == "POST":



        review = request.form.get(
            "review",
            ""
        )



        if review.strip():



            # ---------------------------------
            # Detect Customer Issues
            # ---------------------------------

            customer_issues = detect_customer_issues(
                review
            )




            # ---------------------------------
            # Run BERT Model
            # ---------------------------------

            bert_prediction, bert_confidence, bert_speed = predict_bert(
                review
            )




            # ---------------------------------
            # Run LSTM Model
            # ---------------------------------

            lstm_prediction, lstm_confidence, lstm_speed = predict_lstm(
                review
            )




            # ---------------------------------
            # Business Report
            # (Fixed order)
            # ---------------------------------


            sentiment_result = (
                bert_prediction
                .split("<br>")[0]
                .strip()
            )



            business_report = generate_business_report(

                sentiment_result,

                customer_issues

            )





            # ---------------------------------
            # Business Recommendation
            # ---------------------------------

            recommendations = generate_recommendation(
                review
            )



            recommendation_text = (

                "<br><br>"
                "<b>💼 Business Recommendation</b>"
                "<br>"

            )



            if recommendations:


                for rec in recommendations:

                    recommendation_text += (

                        "• "
                        + rec
                        + "<br>"

                    )



            else:

                recommendation_text += (

                    "No recommendation generated."

                )






            # ---------------------------------
            # Statistics
            # ---------------------------------


            stats = load_stats()



            stats["total_reviews"] += 1




            if "Positive" in bert_prediction:


                stats["positive_reviews"] += 1




            elif "Negative" in bert_prediction:


                stats["negative_reviews"] += 1





            save_stats(stats)






            # ---------------------------------
            # Save History
            # ---------------------------------


            history = load_history()



            history.append({


                "review": review,


                "sentiment":

                    bert_prediction
                    .split("<br>")[0]
                    .strip(),



                "confidence":

                    round(
                        bert_confidence,
                        2
                    ),



                "model":

                    "BERT AI"


            })




            save_history(history)







            # ---------------------------------
            # Best Model Comparison
            # ---------------------------------


            if bert_confidence >= lstm_confidence:


                best_model = (
                    "🏆 Best Model: BERT AI"
                )


                comparison_reason = (

                    "BERT produced higher confidence."

                )



            else:


                best_model = (
                    "🏆 Best Model: Bi-LSTM AI"
                )


                comparison_reason = (

                    "Bi-LSTM produced higher confidence."

                )







            # ---------------------------------
            # Display Prediction
            # ---------------------------------


            prediction = f"""

<div class="model-card">


<h2>
🤖 BERT AI Model
</h2>


{bert_prediction}


</div>




<div class="model-card">


<h2>
🧠 Bi-LSTM AI Model
</h2>


{lstm_prediction}


</div>


"""



    # Load Dashboard Data


    stats = load_stats()



    total_reviews = stats["total_reviews"]


    positive_reviews = stats["positive_reviews"]


    negative_reviews = stats["negative_reviews"]




    history = load_history()



    history = history[-5:]






    return render_template(

        "dashboard.html",


        prediction=prediction,


        review=review,



        bert_prediction=bert_prediction,

        lstm_prediction=lstm_prediction,



        bert_confidence=bert_confidence,

        lstm_confidence=lstm_confidence,



        bert_speed=bert_speed,

        lstm_speed=lstm_speed,



        best_model=best_model,


        comparison_reason=comparison_reason,



        total_reviews=total_reviews,


        positive_reviews=positive_reviews,


        negative_reviews=negative_reviews,



        history=history,



        recommendation_text=recommendation_text,



        business_report=business_report,


        customer_issues=customer_issues

    )






# =====================================================
# Run Application
# =====================================================


if __name__ == "__main__":
    import os

    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=False)