import torch
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np

# get labels first 
class DataProcessor:
    """
    Handles data loading, preprocessing, and conversion to multi-label format.
    """

    def __init__(self, file_path):
        self.file_path = file_path
        self.df = self._load_data()
        self.label_binarizer = None

    def _load_data(self):
        """
        Load dataset from CSV and validate essential columns.
        """
        try:
            df = pd.read_csv(self.file_path)
            required_columns = {"id", "Justification", "Category"}
            if not required_columns.issubset(df.columns):
                raise ValueError(f"Missing required columns: {required_columns - set(df.columns)}")
            return df
        except Exception as e:
            raise RuntimeError(f"Failed to load data: {e}")

    def preprocess_labels(self):
        """
        Cleans and binarizes multi-label classification data.
        """
        df = self.df.dropna(subset=["Category"]).copy()
        df["Category"] = df["Category"].str.lower().str.strip().astype(str).apply(lambda x: x.split("&"))
        df["Category"] = df["Category"].apply(lambda labels: [label.strip() for label in labels])

        # Fit MultiLabelBinarizer
        unique_labels = sorted(set(label for labels in df["Category"] for label in labels))
        self.label_binarizer = MultiLabelBinarizer(classes=unique_labels)
        label_matrix = self.label_binarizer.fit_transform(df["Category"])

        return df.drop(columns=["Category"]).join(pd.DataFrame(label_matrix, columns=unique_labels))


processor = DataProcessor("data.csv")
train_df = processor.preprocess_labels()
id2label = {idx: label for idx, label in enumerate(train_df.drop(columns=["id", "Justification"]).columns)}
print("labels: ", id2label)


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
DF = pd.read_csv(FILE_PATH)


#for testing 
# TEST_SIZE = 5  # Adjust as needed
# df = DF.head(TEST_SIZE)

# real data
df = DF

# Define column that contains the text to classify
TEXT_COLUMN = "justification"  # Change if needed

# Retrieve label names from dataset (assumes columns represent labels)
label_columns = df.columns.tolist()
label_columns.remove(TEXT_COLUMN)  # Remove text column if present
label_columns = [col for col in label_columns if col.lower() != "id"]  # Remove ID if present


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
# Ensure we store all predictions properly
all_predicted_labels = []

with torch.no_grad():
    for batch in dataloader:
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        outputs = model(**batch)
        logits = outputs.logits

        # Convert logits to probabilities
        sigmoid = torch.nn.Sigmoid()
        probs = sigmoid(logits)

        # Get the top-K predictions for each row
        top_k = 2  # Adjust top-K as needed
        top_indices = np.argsort(probs.cpu().numpy(), axis=1)[:, -top_k:]

        batch_predicted_labels = [
            [id2label[i] for i in top_indices[row_idx] if probs[row_idx, i] >= 0.3]  # Ensure minimum probability
            for row_idx in range(probs.shape[0])
        ]

        all_predicted_labels.extend(batch_predicted_labels)  # Store predictions properly

# Ensure the length matches
if len(all_predicted_labels) != len(df):
    raise ValueError(f"Mismatch: {len(all_predicted_labels)} predictions vs {len(df)} rows in DataFrame!")

# Convert predicted labels to a readable format
df["Category"] = ["&".join(labels) if labels else "No Label" for labels in all_predicted_labels]


# Save results
OUTPUT_FILE = "classified_results.csv"
df.to_csv(OUTPUT_FILE, index=False)

