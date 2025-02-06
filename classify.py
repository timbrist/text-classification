"""
This script provides a easy way to train the model: bert-base-uncased in local machine.
But it is leak of flexibility, especially data processing part.
"""

import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments


"""
This class is used 
"""
class Dataprocess:
    def __init__(self, file_path):
        self.df = pd.read_csv(file_path)
    
    """
    This function used for cleaning data for multi-labels classification.
    it will lower the alphabets in the column, cancel the space at the end, 
    split the multi labels that annotate with '&', and finally it will one-hot 
    encoding for each categories. 

    The exmaple dataset in csv file: 
    Id, Text, Category
    30, the car is about to do a right turn, ego car intension: turning
    59, to move around a slow car in its current lane, ego car intension: turning&Interaction:Lead vehicle: Moving Slowly
    86, as the car goes into the other lane, unclassified
    """
    def clean_data(self, column_name="Category"):
        self.df = self.df.dropna(subset=[column_name]).copy()

        self.df[column_name] = self.df[column_name].str.strip()
        self.df[column_name] = self.df[column_name].str.strip().str.lower()
        self.df[column_name] = self.df[column_name].astype(str).apply(lambda x: x.split("&"))
        self.df[column_name] = self.df[column_name].apply(lambda labels: [label.strip() for label in labels])
        
        unique_labels = set(label for labels in self.df["Category"] for label in labels)
        # print(unique_labels)

        mlb = MultiLabelBinarizer(classes=sorted(unique_labels))
        multi_hot_labels = mlb.fit_transform(self.df[column_name])

        label_df = pd.DataFrame(multi_hot_labels, columns=mlb.classes_)

        # Merge with original dataset for reference
        clean_df = self.df.drop(columns=[column_name]).join(label_df)

        return clean_df


# Load the dataset
file_path = '/content/sample_data/data.csv'
df = pd.read_csv(file_path)

# Display the first few rows to understand its structure
print(df.head() )

from sklearn.preprocessing import MultiLabelBinarizer
df_clean = df.dropna(subset=['Category']).copy()

# Ensure any extra spaces are removed
df_clean['Category'] = df_clean['Category'].str.strip()
# Clean up labels: strip spaces and make lowercase consistent
df_clean['Category'] = df_clean['Category'].str.strip().str.lower()

# Step 1: Split multi-label categories and extract unique labels
df_clean["Category"] = df_clean["Category"].astype(str).apply(lambda x: x.split("&"))
df_clean["Category"] = df_clean["Category"].apply(lambda labels: [label.strip() for label in labels])
unique_labels = set(label for labels in df_clean["Category"] for label in labels)
print(unique_labels)
# Step 2: Create a MultiLabelBinarizer to encode labels as multi-hot vectors
mlb = MultiLabelBinarizer(classes=sorted(unique_labels))  # Ensure consistent ordering
multi_hot_labels = mlb.fit_transform(df_clean["Category"])

# Convert to DataFrame for better visualization
label_df = pd.DataFrame(multi_hot_labels, columns=mlb.classes_)

# Merge with original dataset for reference
processed_df = df_clean.drop(columns=["Category"]).join(label_df)

processed_df.to_csv('out.csv')

# Extract text and labels
texts = processed_df["Jusitfication "].tolist()
labels = processed_df.drop(columns=["id", "Jusitfication "]).values  # Multi-hot labels

labels


import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
# Initialize tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Define custom dataset
class MultiLabelDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        item = {key: val.squeeze(0) for key, val in encoding.items()}  # Remove batch dim
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.float)  # Multi-label
        return item

# Split dataset (80% train, 20% test)
train_size = int(0.8 * len(processed_df))
test_size = len(processed_df) - train_size

# Create dataset instances
dataset = MultiLabelDataset(texts, labels, tokenizer)
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

# train_dataset[0]

# Load BERT model for multi-label classification
num_labels = labels.shape[1]
print(num_labels)
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=num_labels, problem_type="multi_label_classification")

training_args = TrainingArguments(
    output_dir="/content/results",             # Output directory
    evaluation_strategy="epoch",        # Evaluate every epoch
    save_strategy="epoch",
    per_device_train_batch_size=8,      # Training batch size
    per_device_eval_batch_size=8,       # Evaluation batch size
    num_train_epochs=5,                 # Number of epochs
    learning_rate=5e-5,  # Starting learning rate
    warmup_steps=500,    # Number of steps to warm up the learning rate
    lr_scheduler_type="linear",  # Type of learning rate scheduler
    seed=42,  # Reproducibility seed
    label_smoothing_factor=0.1,
    weight_decay=0.01,                  # Regularization
    logging_dir="/content/logs",               # Log directory
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    report_to="none"  # Disable WandB logging
)

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import torch.nn as nn
from transformers import Trainer

def compute_metrics(p):
    sigmoid = lambda x: 1 / (1 + np.exp(-x))  # Sigmoid activation
    preds = sigmoid(p.predictions) > 0.5  # Convert logits to binary (threshold = 0.5)
    labels = p.label_ids

    acc = accuracy_score(labels, preds)  # Multi-label accuracy

    # Compute F1-score with zero_division=1 to avoid warnings
    f1_micro = f1_score(labels, preds, average="micro", zero_division=1)
    f1_macro = f1_score(labels, preds, average="macro", zero_division=1)
    f1_samples = f1_score(labels, preds, average="samples", zero_division=1)

    return {
        "accuracy": acc,
        "f1_micro": f1_micro,
        "f1_macro": f1_macro,
        "f1_samples": f1_samples
    }



class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):  # Accept extra arguments
        labels = inputs.pop("labels")  # Remove labels from inputs
        outputs = model(**inputs)  # Forward pass
        logits = outputs.logits  # Extract logits

        # Define loss function for multi-label classification
        loss_fn = nn.BCEWithLogitsLoss()
        loss = loss_fn(logits, labels)

        return (loss, outputs) if return_outputs else loss


# #testing small set first just to check if everything is runing ok
# small_train_dataset = torch.utils.data.Subset(train_dataset, range(100))  # Use only 100 samples
# small_test_dataset = torch.utils.data.Subset(test_dataset, range(20))  # Use only 20 samples
# trainer = CustomTrainer(
#     model=model,
#     args=training_args,
#     train_dataset=small_train_dataset,
#     eval_dataset=small_test_dataset,
#     tokenizer=tokenizer,
#     compute_metrics=compute_metrics
# )


trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# Train the model

trainer.train()

# Evaluate the model
eval_results = trainer.evaluate()
print("Evaluation Results:", eval_results)

# Inference function
def predict(text):
    model.eval()
    encoding = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=512)
    encoding = {k: v.to(trainer.model.device) for k, v in encoding.items()}
    
    with torch.no_grad():
        outputs = trainer.model(**encoding)
    
    logits = outputs.logits
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(logits.squeeze().cpu())
    predictions = np.zeros(probs.shape)
    predictions[np.where(probs >= 0.5)] = 1
    
    # Map predictions back to labels
    id2label = {idx: label for idx, label in enumerate(df.drop(columns=["id", "Jusitfication"]).columns)}
    predicted_labels = [id2label[idx] for idx, label in enumerate(predictions) if label == 1.0]
    
    return predicted_labels

# Example prediction
example_text = "to move around a slow car in its current lane."
predicted_labels = predict(example_text)
print("Predicted Labels:", predicted_labels)
