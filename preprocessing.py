import pandas as pd
import re
import nltk
import matplotlib.pyplot as plt
from wordcloud import WordCloud

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Load dataset
data = pd.read_csv("dataset/Womens Clothing E-Commerce Reviews.csv")

# Remove rows with missing review text
data = data.dropna(subset=["Review Text"])

# Create stopwords and lemmatizer
stop_words = set(stopwords.words("english"))

# Keep important negation words
stop_words.discard("not")
stop_words.discard("no")
stop_words.discard("never")

lemmatizer = WordNetLemmatizer()

# Function to clean text
def clean_text(text):

    text = text.lower()

    # Convert positive emojis into words
    text = text.replace("😍", " love ")
    text = text.replace("😊", " happy ")
    text = text.replace("😁", " happy ")
    text = text.replace("😄", " happy ")
    text = text.replace("❤️", " love ")
    text = text.replace("❤", " love ")
    text = text.replace("👍", " good ")

    # Convert negative emojis into words
    text = text.replace("😞", " sad ")
    text = text.replace("😢", " sad ")
    text = text.replace("😭", " sad ")
    text = text.replace("😡", " angry ")
    text = text.replace("👎", " bad ")

    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text)

    # Split into words
    words = text.split()

    # Remove stopwords
    words = [word for word in words if word not in stop_words]

    # Lemmatize words
    words = [lemmatizer.lemmatize(word) for word in words]

    return " ".join(words).strip()

# Apply cleaning
data["Clean Review"] = data["Review Text"].apply(clean_text)

# Display original and cleaned reviews
for i in range(5):
    print("\nOriginal Review:")
    print(data["Review Text"].iloc[i])

    print("\nClean Review:")
    print(data["Clean Review"].iloc[i])

    print("-" * 70)
    
    # Save cleaned dataset
data.to_csv("dataset/clean_reviews.csv", index=False)

print("\nCleaned dataset saved successfully!")

# Rating Distribution Chart

plt.figure(figsize=(8,5))

data["Rating"].value_counts().sort_index().plot(
    kind="bar"
)

plt.title("Customer Rating Distribution")
plt.xlabel("Rating")
plt.ylabel("Number of Reviews")

plt.savefig("results/rating_distribution.png")
plt.show()

# ----------------------------
# Word Cloud
# ----------------------------

# Combine all cleaned reviews into one string
text = " ".join(data["Clean Review"])

# Generate Word Cloud
wordcloud = WordCloud(
    width=1000,
    height=500,
    background_color="white"
).generate(text)

# Display Word Cloud
plt.figure(figsize=(12,6))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.title("Word Cloud of Customer Reviews")

# Save the image
plt.savefig("results/wordcloud.png")

# Show the image
plt.show()

# ----------------------------
# Recommended vs Not Recommended
# ----------------------------

plt.figure(figsize=(6,5))

data["Recommended IND"].value_counts().plot(
    kind="bar"
)

plt.title("Recommended vs Not Recommended")
plt.xlabel("Recommendation")
plt.ylabel("Number of Reviews")

plt.xticks([0,1], ["Not Recommended", "Recommended"], rotation=0)

plt.savefig("results/recommendation_chart.png")

plt.show()