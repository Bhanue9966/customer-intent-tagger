# src/model_trainer.py

import os
import torch
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score

from transformers import (
    AutoModelForSequenceClassification, 
    TrainingArguments, 
    Trainer # Removed IntervalStrategy from imports
)
from src.data_processor import DataProcessor 

# Define a custom PyTorch Dataset class
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = {k: torch.tensor(v) for k, v in encodings.items()}
        self.labels = torch.tensor(labels, dtype=torch.float)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.labels)

# --- Metric Computation Function ---
def compute_metrics(p):
    """Computes a set of relevant metrics for multi-label/multi-task classification."""
    y_pred_proba = torch.sigmoid(torch.tensor(p.predictions)).numpy()
    y_pred = (y_pred_proba > 0.5).astype(int)
    y_true = p.label_ids

    # Calculate metrics
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)
    
    try:
        roc_auc = roc_auc_score(y_true, y_pred_proba, average='macro')
    except ValueError:
        roc_auc = 0.0 

    return {
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'roc_auc': roc_auc,
    }


def train_and_save_model(data_file_path, output_dir='models/bert_classifier'):
    """Initializes model, prepares data, trains the model, and saves all artifacts."""
    
    # --- 1. Data Preparation ---
    processor = DataProcessor(model_name='distilbert-base-uncased')
    train_encodings, val_encodings = processor.get_train_test_split(data_file_path)
    
    train_dataset = CustomDataset(train_encodings, train_encodings['labels'])
    val_dataset = CustomDataset(val_encodings, val_encodings['labels'])

    num_labels = len(processor.ALL_LABELS)

    # --- 2. Model Initialization ---
    model = AutoModelForSequenceClassification.from_pretrained(
        'distilbert-base-uncased', 
        num_labels=num_labels
    )
    
    # --- 3. Training Arguments ---
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,                     
        per_device_train_batch_size=16,         
        per_device_eval_batch_size=16,          
        warmup_steps=500,                       
        weight_decay=0.01,                      
        logging_dir='./logs',                   
        logging_steps=50,

        # FINAL FIX: Remove evaluation logic to bypass old version TypeErrors.
        # We set save_steps high so it only saves the final model after all epochs are done.
        save_steps=2000, 
        
        # The following arguments are REMOVED entirely to prevent TypeErrors:
        # evaluation_strategy, save_strategy, load_best_model_at_end, metric_for_best_model
    )

    # --- 4. Trainer Setup & Training ---
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )
    
    print("\n--- Starting Model Training ---")
    trainer.train()

    # --- 5. Evaluation and Saving ---
    final_save_path = os.path.join(output_dir, "final_model")
    trainer.save_model(final_save_path)
    processor.tokenizer.save_pretrained(final_save_path)
    
    # Save the ALL_LABELS list for the prediction script
    with open(os.path.join(final_save_path, 'all_labels.txt'), 'w') as f:
        for label in processor.ALL_LABELS:
            f.write(f"{label}\n")
    print(f"\nModel and artifacts successfully saved to: {final_save_path}")


if __name__ == '__main__':
    # Define the path to your RAW data file in the project root
    FILE_PATH = 'Bitext_Sample_Customer_Service_Training_Dataset.csv'
    
    # Run the training process
    train_and_save_model(FILE_PATH)