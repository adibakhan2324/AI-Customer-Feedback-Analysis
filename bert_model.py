import pandas as pd
import torch

from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import Trainer, TrainingArguments

# Load cleaned dataset
data = pd.read_csv("dataset/clean_reviews.csv")

# Select review text and labels
X = data["Clean Review"]
y = data["Recommended IND"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

# Load BERT tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

print("Tokenizer loaded successfully!")

# Tokenize training data
train_encodings = tokenizer(
    X_train.tolist(),
    truncation=True,
    padding=True,
    max_length=128
)

# Tokenize testing data
test_encodings = tokenizer(
    X_test.tolist(),
    truncation=True,
    padding=True,
    max_length=128
)

print("Reviews tokenized successfully!")

# Create PyTorch Dataset
class ReviewDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels.tolist()

    def __getitem__(self, idx):
        item = {}

        for key, value in self.encodings.items():
            item[key] = torch.tensor(value[idx])

        item["labels"] = torch.tensor(self.labels[idx])

        return item

    def __len__(self):
        return len(self.labels)
    
    # Create training and testing datasets
train_dataset = ReviewDataset(train_encodings, y_train)
test_dataset = ReviewDataset(test_encodings, y_test)

print("PyTorch datasets created successfully!")

# Load BERT model
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=2
)

print("BERT model loaded successfully!")

# Training configuration
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    logging_dir="./logs",
    logging_steps=100
)

print("Training arguments created successfully!")

# Create Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

print("Trainer created successfully!")

# Train the model
print("Starting BERT training...")

trainer.train()

print("BERT model training completed!")

# Evaluate the model
results = trainer.evaluate()

print("Evaluation Results:")
print(results)

# Save the trained model
model.save_pretrained("models/bert_model")
tokenizer.save_pretrained("models/bert_model")

print("BERT model saved successfully!")