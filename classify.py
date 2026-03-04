import os
import logging
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, random_split
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments,
    EvalPrediction,
    BertConfig
)
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from typing import Dict, Any, Tuple, List

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
MODEL_NAME = "bert-large-uncased"
TRAIN_TEST_SPLIT = 0.8
BATCH_SIZE = 32
NUM_EPOCHS = 20
LEARNING_RATE = 9e-5
PREDICT_THRESHOLD = 0.7
WEIGHT_DECAY = 0.5
DROPOUT = 0.1
WARMUP_RATIO = 0.1

LOG_DIR = "./log"
OUTPUT_DIR = "./results"
DATA_PATH = "cleaned_data.csv"

# Ensure directories exist
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Initialize tokenizer
TOKENIZER = BertTokenizer.from_pretrained(MODEL_NAME)


class DataProcessor:
    """Handles data loading, preprocessing, and conversion to multi-label format."""
    
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.df = self._load_data()
        self.label_binarizer = None

    def _load_data(self) -> pd.DataFrame:
        """Load dataset from CSV and validate essential columns."""
        try:
            df = pd.read_csv(self.file_path)
            required_columns = {"id", "Justification", "Category"}
            if not required_columns.issubset(df.columns):
                missing = required_columns - set(df.columns)
                raise ValueError(f"Missing required columns: {missing}")
            return df
        except Exception as e:
            raise RuntimeError(f"Failed to load data: {e}")

    def preprocess_labels(self) -> pd.DataFrame:
        """Cleans and binarizes multi-label classification data."""
        df = self.df.dropna(subset=["Category"]).copy()
        # Split on '&', lower-case, strip extra spaces, and clean each label
        df["Category"] = (
            df["Category"]
            .str.lower()
            .str.strip()
            .apply(lambda x: [label.strip() for label in x.split("&")])
        )
        # Fit MultiLabelBinarizer
        unique_labels = sorted({label for labels in df["Category"] for label in labels})
        self.label_binarizer = MultiLabelBinarizer(classes=unique_labels)
        label_matrix = self.label_binarizer.fit_transform(df["Category"])
        return df.drop(columns=["Category"]).join(pd.DataFrame(label_matrix, columns=unique_labels))

    def get_datasets(self, text_column: str = "Justification") -> Tuple[pd.DataFrame, Dataset, Dataset, int]:
        """Splits data into training and testing sets."""
        df = self.preprocess_labels()
        texts = df[text_column].tolist()
        labels = df.drop(columns=["id", text_column]).values
        dataset = MultiLabelDataset(texts, labels, TOKENIZER)
        train_size = int(TRAIN_TEST_SPLIT * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
        num_labels = labels.shape[1]
        return df, train_dataset, test_dataset, num_labels


class MultiLabelDataset(Dataset):
    """Custom Dataset for multi-label classification."""
    
    def __init__(self, texts: List[str], labels: Any, tokenizer, max_length: int = 512):
        self.labels = labels
        self.encodings = tokenizer(
            texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt"
        )

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels": torch.tensor(self.labels[idx], dtype=torch.float)
        }



class MultiLabelTrainer:
    """
    Manages model training and evaluation using Hugging Face's Trainer API.
    """
    
    def __init__(self, num_labels: int, train_dataset: Dataset, test_dataset: Dataset):
        self.num_labels = num_labels


        self.model = BertForSequenceClassification.from_pretrained(
            MODEL_NAME,
            num_labels=num_labels,
            problem_type="multi_label_classification",
            hidden_dropout_prob=DROPOUT,           # Dropout probability for fully connected layers
            attention_probs_dropout_prob=DROPOUT   # Dropout probability for attention weights
        )


        warmup_ratio = WARMUP_RATIO  # Use 10% of total steps as warmup
        num_training_steps = (len(train_dataset) * NUM_EPOCHS) // BATCH_SIZE
        warmup_steps = int(warmup_ratio * num_training_steps)
        print(f"the warmup steps: {warmup_steps}")

        training_args = TrainingArguments(
            output_dir=OUTPUT_DIR,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            per_device_train_batch_size=BATCH_SIZE,
            per_device_eval_batch_size=BATCH_SIZE,
            num_train_epochs=NUM_EPOCHS,
            learning_rate=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
            warmup_steps=warmup_steps,
            lr_scheduler_type="linear",
            seed=42,
            logging_steps=5,
            logging_dir=LOG_DIR,
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            report_to="none"
        )

        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            tokenizer=TOKENIZER,
            compute_metrics=self.compute_metrics
        )

    @staticmethod
    def compute_metrics(p: EvalPrediction) -> Dict[str, float]:
        """Computes multi-label classification metrics."""
        sigmoid = torch.nn.Sigmoid()
        probs = sigmoid(torch.Tensor(p.predictions))
        y_pred = (probs >= PREDICT_THRESHOLD).int().numpy()
        f1 = f1_score(p.label_ids, y_pred, average="micro")
        try:
            roc_auc = roc_auc_score(p.label_ids, y_pred, average="micro")
        except ValueError:
            roc_auc = 0.5  # Fallback if computation fails
        accuracy = accuracy_score(p.label_ids, y_pred)
        return {"f1": f1, "roc_auc": roc_auc, "accuracy": accuracy}

    def train(self):
        """Runs the training loop."""
        self.trainer.train()
        eval_results = self.trainer.evaluate()
        logger.info("Evaluation Results: %s", eval_results)

    def predict(self, df: pd.DataFrame, example_text: str = "to move around a slow car in its current lane.", threshold: float = PREDICT_THRESHOLD):
        """Runs inference on a single example."""
        encoding = TOKENIZER(example_text, return_tensors="pt")
        encoding = {k: v.to(self.model.device) for k, v in encoding.items()}
        outputs = self.trainer.model(**encoding)
        probs = torch.sigmoid(outputs.logits.squeeze().cpu())
        predictions = (probs >= threshold).numpy().astype(int)
        id2label = {idx: label for idx, label in enumerate(df.drop(columns=["id", "Justification"]).columns)}
        predicted_labels = [id2label[idx] for idx, val in enumerate(predictions) if val == 1]
        logger.info("Predicted labels: %s", predicted_labels)


def main():
    processor = DataProcessor(DATA_PATH)
    df, train_dataset, test_dataset, num_labels = processor.get_datasets()
    trainer = MultiLabelTrainer(num_labels, train_dataset, test_dataset)
    trainer.train()
    trainer.predict(df)


if __name__ == "__main__":
    main()
