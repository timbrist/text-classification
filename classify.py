"""
This script trains a BERT-based multi-label text classifier using PyTorch and Hugging Face Transformers.
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, EvalPrediction
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
import os

# Configuration
MODEL_NAME = "bert-base-uncased"
TRAIN_TEST_SPLIT = 0.8
BATCH_SIZE = 8
NUM_EPOCHS = 20
LEARNING_RATE = 1e-5
PREDICT_THRESHOLD = 0.7
TOKENIZER = BertTokenizer.from_pretrained(MODEL_NAME)



# File paths
DATA_PATH = "data.csv"
LOG_DIR = "./log"
OUTPUT_DIR = "./results"


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

    def get_datasets(self, text_column="Justification"):
        """
        Splits data into training and testing sets.
        """
        df = self.preprocess_labels()
        texts, labels = df[text_column].tolist(), df.drop(columns=["id", text_column]).values

        dataset = MultiLabelDataset(texts, labels, TOKENIZER)
        train_size = int(TRAIN_TEST_SPLIT * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

        return df, train_dataset, test_dataset, labels.shape[1]


class MultiLabelDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.labels = labels
        self.encodings = tokenizer.batch_encode_plus(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels": torch.tensor(self.labels[idx], dtype=torch.float)
        }

    

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE_loss = nn.BCEWithLogitsLoss()(inputs, targets)
        probs = torch.sigmoid(inputs)
        focal_weight = self.alpha * (1 - probs) ** self.gamma
        return (focal_weight * BCE_loss).mean()

class MultiLabelTrainer:
    """
    Manages model training and evaluation using Hugging Face's Trainer API.
    """

    def __init__(self, num_labels, train_dataset, test_dataset):
        self.num_labels = num_labels
        self.model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=num_labels, problem_type="multi_label_classification")
        self.model.config.problem_type = "multi_label_classification"
        self.model.loss_fn = FocalLoss()

        warmup_ratio = 0.1  # Use 10% of total steps as warmup
        num_training_steps = len(train_dataset) * NUM_EPOCHS // BATCH_SIZE
        warmup_steps = int(warmup_ratio * num_training_steps)


        self.trainer = Trainer(
            model=self.model,
            args=TrainingArguments(
                output_dir=OUTPUT_DIR,
                evaluation_strategy="epoch",
                save_strategy="steps",
                save_steps=1000,  # Save every 1000 steps
                save_total_limit=1,  # Keep only the best model
                load_best_model_at_end=True,
                per_device_train_batch_size=BATCH_SIZE,
                per_device_eval_batch_size=BATCH_SIZE,
                num_train_epochs=NUM_EPOCHS,
                learning_rate=LEARNING_RATE,
                weight_decay=0.1,
                warmup_steps=warmup_steps,
                lr_scheduler_type="linear",
                seed=42,
                logging_dir=LOG_DIR,
                # logging_steps=10,
                metric_for_best_model="accuracy",
                report_to="none"
            ),
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            tokenizer=TOKENIZER,
            compute_metrics=self.compute_metrics
        )

    @staticmethod
    def compute_metrics(p: EvalPrediction):
        sigmoid = nn.Sigmoid()
        probs = sigmoid(torch.tensor(p.predictions))
        y_pred = (probs >= PREDICT_THRESHOLD).int().numpy()

        return {
            "f1": f1_score(p.label_ids, y_pred, average='micro'),
            "roc_auc": roc_auc_score(p.label_ids, y_pred, average="macro"),
            "accuracy": accuracy_score(p.label_ids, y_pred)
        }

    def train(self):
        """
        Runs the training loop.
        """
        self.trainer.train()
        eval_results = self.trainer.evaluate()
        print("Evaluation Results:", eval_results)

    def predict(self, df, example_text="to move around a slow car in its current lane."):
        """
        Runs inference on a single example.
        """
        encoding = TOKENIZER(example_text, return_tensors="pt")
        outputs = self.trainer.model(**{k: v.to(self.model.device) for k, v in encoding.items()})

        sigmoid = nn.Sigmoid()
        probs = sigmoid(outputs.logits.squeeze().cpu())
        predictions = (probs >= PREDICT_THRESHOLD).numpy().astype(int)

        id2label = {idx: label for idx, label in enumerate(df.drop(columns=["id", "Justification"]).columns)}
        predicted_labels = [id2label[idx] for idx, val in enumerate(predictions) if val == 1]
        print("Predicted labels:", predicted_labels)


def main():
    processor = DataProcessor(DATA_PATH)
    df, train_dataset, test_dataset, num_labels = processor.get_datasets()

    trainer = MultiLabelTrainer(num_labels, train_dataset, test_dataset)
    trainer.train()
    trainer.predict(df)


if __name__ == "__main__":
    main()
