import torch
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
import numpy as np

# Path to trained model
MODEL_PATH = "results/epoch20-acc081"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("Device is ", DEVICE)

torch.cuda.empty_cache()

# Load tokenizer and model
tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
model.to(DEVICE)
model.eval()  # Set model to evaluation mode

# Load the uploaded file
FILE_PATH = "groundtruth_trainset.csv"
df = pd.read_csv(FILE_PATH)

# Define column that contains the text to classify
TEXT_COLUMN = "justification"  # Change this if the column name is different

# Tokenize text
encoded_inputs = tokenizer(
    df[TEXT_COLUMN].tolist(),
    padding=True,
    truncation=True,
    max_length=512,
    return_tensors="pt"
)

# Move tensors to GPU if available
encoded_inputs = {key: value.to(DEVICE) for key, value in encoded_inputs.items()}

# Run inference
with torch.no_grad():
    outputs = model(**encoded_inputs)
    logits = outputs.logits

# Convert logits to probabilities
sigmoid = torch.nn.Sigmoid()
probs = sigmoid(logits)
predictions = (probs >= 0.7).int().cpu().numpy()  # Adjust threshold if needed (e.g., 0.7)

# Load label names (assuming the model has stored them in order)
label_names = model.config.id2label if hasattr(model.config, "id2label") else [f"label_{i}" for i in range(logits.shape[1])]

# Convert predictions to label names
predicted_labels = [
    [label_names[i] for i, pred in enumerate(pred_row) if pred == 1]
    for pred_row in predictions
]

# Add predictions to DataFrame
df["Predicted Labels"] = ["; ".join(labels) for labels in predicted_labels]

# Save results
OUTPUT_FILE = "classified_results.csv"
df.to_csv(OUTPUT_FILE, index=False)


