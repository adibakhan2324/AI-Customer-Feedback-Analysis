import pandas as pd

data = pd.read_csv("dataset/clean_reviews.csv")

print(data["Recommended IND"].value_counts())