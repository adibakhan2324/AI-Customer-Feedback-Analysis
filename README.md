# 🤖 CustomerPulse AI

# AI-Based Customer Feedback Analysis System for Business Decision Making

<p align="center">
<img src="screenshots/final_ai_dashboard.png" width="900">
</p>

<p align="center">

An intelligent NLP-based customer feedback analysis platform that uses Deep Learning models such as **BERT and Bi-LSTM** to understand customer opinions, detect issues, analyze sentiment, and generate AI-powered business recommendations.

</p>

---

# 📌 Project Overview

Customer feedback plays an important role in improving products and services. However, organizations receive thousands of reviews every day, making manual analysis difficult, time-consuming, and inefficient.

**CustomerPulse AI** is an Artificial Intelligence and Natural Language Processing based system that automatically analyzes customer reviews and transforms unstructured feedback into meaningful business insights.

The system performs:

- Sentiment classification
- Customer issue detection
- Language pattern analysis
- AI-generated business reports
- Recommendation generation
- Interactive dashboard visualization

The project helps businesses make faster and smarter decisions using AI-driven feedback analytics.

---

# 🎯 Problem Statement

Businesses collect large amounts of customer reviews from different platforms.

Traditional manual analysis methods face problems such as:

- Large volume of feedback
- Slow decision-making
- Difficulty identifying repeated complaints
- Human bias in analysis
- Lack of real-time insights

Therefore, an automated AI system is required to analyze customer feedback efficiently and provide actionable insights.

---

# 💡 Project Motivation

The main motivation behind this project is to bridge the gap between:

**Customer Opinions → AI Analysis → Business Decisions**

By applying Deep Learning and NLP techniques, businesses can understand:

- What customers like
- What problems customers face
- Why customers are dissatisfied
- What improvements are required

---

# 🎯 Objectives

The objectives of CustomerPulse AI are:

✔ Automatically analyze customer reviews using Artificial Intelligence

✔ Classify feedback into positive, negative, and mixed sentiment

✔ Compare transformer-based and sequential deep learning models

✔ Detect important customer issues from reviews

✔ Generate AI-based business recommendations

✔ Provide visual analytics for decision-making

---

# ⭐ Key Features

## 🧠 AI Sentiment Analysis

The system predicts customer sentiment using Deep Learning models.

Supported categories:

😊 Positive Review

😞 Negative Review

😐 Mixed Sentiment


The system provides:

- Sentiment prediction
- Confidence score
- Confidence level
- Model used

---

# 🔍 Customer Issue Detection

The AI system automatically identifies common customer problems.

Detected issues include:

🚚 Delivery Problems

☎️ Customer Support Issues

📦 Product Quality Problems

💳 Service Related Issues


This helps businesses understand the main reasons behind customer dissatisfaction.

---

# 🧠 AI Language Analysis

The system analyzes customer language patterns to identify:

- Emotional tone
- Important keywords
- Complaint patterns
- Customer concerns

---

# 📊 AI Business Report Generation

CustomerPulse AI automatically generates business reports containing:

### Customer Sentiment Summary

### Detected Issues

### Business Impact

### Recommended Actions


Example:

```
Detected Issue:
Delivery Delay

Business Impact:
Delayed delivery may reduce customer satisfaction.

Recommended Action:
Improve logistics management and delivery tracking.
```

---

# 🏗️ System Architecture

```
              Customer Reviews

                    |
                    ↓

          Data Preprocessing Layer

                    |
                    ↓

          NLP Processing Pipeline

                    |
                    ↓

        ┌─────────────────────┐
        │                     │
        ↓                     ↓

      BERT Model          Bi-LSTM Model

        │                     │
        └──────────┬──────────┘

                   ↓

        Sentiment Classification

                   ↓

        Customer Issue Detection

                   ↓

        AI Business Report

                   ↓

        Interactive Dashboard
```

---

# 🔬 Deep Learning Models

# 1. BERT Model

## Bidirectional Encoder Representations from Transformers


BERT is a transformer-based language model that understands the context of words by analyzing the complete sentence.

### Advantages:

- Better contextual understanding
- Handles complex customer sentences
- Improved sentiment classification


---

# 2. Bi-LSTM Model

## Bidirectional Long Short-Term Memory Network


Bi-LSTM is a recurrent neural network that learns sequential patterns from customer feedback.

### Advantages:

- Captures word relationships
- Learns review patterns
- Effective for text classification


---

# ⚖️ BERT vs Bi-LSTM Comparison

| Feature | BERT | Bi-LSTM |
|-|-|-|
| Architecture | Transformer | Recurrent Neural Network |
| Context Understanding | Very High | Medium |
| Training Speed | Slower | Faster |
| Memory Requirement | Higher | Lower |
| Text Representation | Contextual Embedding | Sequential Learning |
| Performance | High Accuracy | Competitive |

---

# 📂 Dataset Information

Dataset Used:

## Women's Clothing E-Commerce Reviews Dataset


The dataset contains customer reviews with information including:

- Review text
- Product information
- Customer details
- Ratings


## Data Preprocessing Steps:

1. Data cleaning

2. Removing missing values

3. Text normalization

4. Tokenization

5. Label preparation

6. Model-ready data generation


---

# ⚙️ Technology Stack


## Programming Language

- Python


## Artificial Intelligence

- BERT
- Bi-LSTM
- PyTorch
- TensorFlow
- Keras
- Scikit-learn


## Natural Language Processing

- Hugging Face Transformers
- Tokenization
- Text preprocessing


## Backend

- Flask


## Frontend

- HTML
- CSS
- JavaScript


## Visualization

- Chart.js
- Matplotlib


---

# 📁 Project Structure

```
AI-Customer-Feedback-Analysis

│
├── app.py
├── preprocessing.py
├── bert_model.py
├── lstm_model.py
├── business_report.py
├── business_recommendation.py
├── customer_issue_detection.py
├── language_rules.py
│
├── templates/
│
├── static/
│
├── screenshots/
│
└── requirements.txt

```

---

# 🖥️ Application Screenshots


## 🏠 Home Dashboard

![Home Dashboard](screenshots/home_dashboard.png)


## 📝 Customer Review Input

![Review Input](screenshots/review_input.png)


## 🤖 Sentiment Analysis

![Sentiment Analysis](screenshots/sentiment_analysis.png)


## 😊 Positive Prediction

![Positive Result](screenshots/sentiment_result_Positive.png)


## 😞 Negative Prediction

![Negative Result](screenshots/sentiment_result_Negative.png)


## 😐 Mixed Sentiment

![Mixed Sentiment](screenshots/mixed_sentiment.png)


## 📈 Analytics Dashboard

![Analytics](screenshots/analytics_dashboard.png)


## ⚖️ Model Comparison

![Model Comparison](screenshots/model_comparison.png)


## 🎯 Confidence Score

![Confidence](screenshots/confidence_score.png)


## 📚 Review History

![History](screenshots/review_history.png)


## ℹ️ About Section

![About](screenshots/about_section.png)


---

# 🚀 Installation and Execution


Clone repository:

```bash
git clone https://github.com/adibakhan2324/AI-Customer-Feedback-Analysis.git
```


Navigate:

```bash
cd AI-Customer-Feedback-Analysis
```


Create environment:

```bash
python -m venv .venv
```


Activate:

```bash
.venv\Scripts\activate
```


Install dependencies:

```bash
pip install -r requirements.txt
```


Run application:

```bash
python app.py
```


Open:

```
http://127.0.0.1:5000
```

---

# 📈 Business Applications

CustomerPulse AI can support:

- E-commerce companies
- Customer support departments
- Product management teams
- Market research organizations


Applications:

✔ Customer satisfaction monitoring

✔ Complaint analysis

✔ Product improvement

✔ Service quality improvement

✔ Business decision support


---

# ⚠️ Limitations

- Currently focused on English reviews
- Performance depends on dataset quality
- Requires retraining for different domains
- Large AI models require computational resources


---

# 🔮 Future Enhancements

Future improvements include:

- Real-time customer feedback monitoring
- Multilingual sentiment analysis
- Voice feedback analysis
- Generative AI business assistant
- Mobile application
- Cloud deployment
- Real-time dashboard integration


---

# 👩‍💻 Author

## Adiba Khan

B.Tech Computer Science Engineering


---

# ⭐ Project Status

✅ Completed

Developed using Artificial Intelligence, Deep Learning, and Natural Language Processing for intelligent customer feedback analysis.