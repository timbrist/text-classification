import torch
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, Dataset

# Path to trained model
MODEL_PATH = "results/epoch20-acc081"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16  # Adjust based on GPU memory
PREDICT_THRESHOLD = 0.7  # Change this for better precision-recall balance

# Load tokenizer and model
tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
model.to(DEVICE)
model.eval()  # Set model to evaluation mode

# Load the uploaded file
FILE_PATH = "groundtruth_trainset.csv"
df = pd.read_csv(FILE_PATH)

# Define column that contains the text to classify
TEXT_COLUMN = "justification"  # Change if needed

# Retrieve label names from dataset (assumes columns represent labels)
label_columns = df.columns.tolist()
label_columns.remove(TEXT_COLUMN)  # Remove text column if present
label_columns = [col for col in label_columns if col.lower() != "id"]  # Remove ID if present

# Ensure model has label mapping (otherwise, use dataset labels)
if hasattr(model.config, "id2label"):
    label_names = [model.config.id2label[i] for i in range(len(model.config.id2label))]
else:
    label_names = label_columns  # Use dataset labels as a fallback

# Custom Dataset class to handle batching
class JustificationDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=512):
        self.texts = texts
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
        return {key: val.squeeze(0) for key, val in encoding.items()}  # Remove batch dim

# Create Dataset and DataLoader
dataset = JustificationDataset(df[TEXT_COLUMN].tolist(), tokenizer)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

# Run inference in batches
all_predictions = []
with torch.no_grad():
    for batch in dataloader:
        batch = {k: v.to(DEVICE) for k, v in batch.items()}  # Move batch to GPU
        outputs = model(**batch)
        logits = outputs.logits

        # Convert logits to probabilities
        sigmoid = torch.nn.Sigmoid()
        probs = sigmoid(logits)

        # Apply threshold (default 0.5)
        predictions = (probs >= PREDICT_THRESHOLD).int().cpu().numpy()
        all_predictions.extend(predictions)

# Convert predictions to label names
predicted_labels = [
    [label_names[i] for i, pred in enumerate(pred_row) if pred == 1]
    for pred_row in all_predictions
]

# Add predictions to DataFrame
df["Category"] = ["; ".join(labels) if labels else "No Label" for labels in predicted_labels]

# Save results
OUTPUT_FILE = "classified_results.csv"
df.to_csv(OUTPUT_FILE, index=False)

