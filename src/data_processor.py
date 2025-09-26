# src/data_processor.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer

class DataProcessor:
    """
    Loads, processes, and prepares the dataset for a multi-task (intent + tags) 
    transformer model.
    """
    def __init__(self, model_name='distilbert-base-uncased', max_length=128):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_length = max_length
        self.INTENT_LABELS = None
        self.TAG_LABELS = None
        self.ALL_LABELS = None

    def load_data(self, file_path):
        """Loads the raw CSV file and cleans column names."""
        df = pd.read_csv(file_path)
        # FIX: Strip whitespace from all column names (fixes KeyError)
        df.columns = df.columns.str.strip()
        print("Columns found after cleaning:", df.columns.tolist())
        return df

    def create_labels(self, df):
        """Identifies all unique intents and single tags."""
        
        # 1. Multi-Class Intent Labels
        self.INTENT_LABELS = sorted(df['intent'].unique().tolist())
        
        # 2. Multi-Label Tag Labels (Based on the FE from the notebook)
        # We assume the original file contained 'tags' for this to work
        all_tags_list = "".join(df['tags'].dropna().unique())
        unique_single_tags = sorted(list(set(char for char in all_tags_list if char.isalpha())))
        self.TAG_LABELS = [f'tag_{t}' for t in unique_single_tags]
        
        # 3. Combined All Labels
        self.ALL_LABELS = self.INTENT_LABELS + self.TAG_LABELS
        
        return unique_single_tags

    def preprocess(self, df):
        """Performs all necessary data transformations (One-Hot Encoding)."""
        
        unique_single_tags = self.create_labels(df)
        
        # --- Feature Engineering for Tags (Multi-Label) ---
        for tag in unique_single_tags:
            # Assumes the original 'tags' column exists
            df[f'tag_{tag}'] = df['tags'].apply(lambda x: 1 if tag in str(x) else 0)
            
        # --- One-Hot Encoding for Intent (Multi-Class) ---
        for intent in self.INTENT_LABELS:
            df[intent] = df['intent'].apply(lambda x: 1 if x == intent else 0)
            
        # Combine the target columns into a single list/array
        target_columns = self.INTENT_LABELS + self.TAG_LABELS
        df['labels'] = df[target_columns].values.tolist()
        
        return df[['utterance', 'labels']]

    def tokenize_data(self, df):
        """Tokenizes the utterances and prepares the final dataset."""
        tokenized_inputs = self.tokenizer(
            df['utterance'].tolist(),
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='np' # Return NumPy arrays
        )
        labels = np.array(df['labels'].tolist(), dtype=float)
        
        dataset = {
            'input_ids': tokenized_inputs['input_ids'],
            'attention_mask': tokenized_inputs['attention_mask'],
            'labels': labels
        }
        return dataset

    def get_train_test_split(self, file_path, test_size=0.1, random_state=42):
        """Executes the full pipeline: Load -> Process -> Tokenize -> Split."""
        
        df = self.load_data(file_path)
        df_processed = self.preprocess(df)
        
        # Split data *before* tokenization 
        train_df, test_df = train_test_split(
            df_processed, 
            test_size=test_size, 
            random_state=random_state, 
            shuffle=True
        )

        train_dataset = self.tokenize_data(train_df)
        test_dataset = self.tokenize_data(test_df)
        
        return train_dataset, test_dataset