# src/predictor.py

import os
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class IntentTaggerPredictor:
    """
    Handles loading the fine-tuned model and making predictions 
    for Intent (Multi-Class) and Tags (Multi-Label).
    """
    def __init__(self, model_path, max_length=128):
        self.model_path = model_path
        self.max_length = max_length
        self.tokenizer = None
        self.model = None
        self.all_labels = None
        self.intent_labels = None
        self.tag_labels = None
        self.load_artifacts()

    def load_artifacts(self):
        """Loads the tokenizer, model, and labels list."""
        print(f"Loading artifacts from {self.model_path}...")
        
        # Load Tokenizer and Model
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
        self.model.eval() # Set model to evaluation mode
        
        # Load Labels
        labels_file = os.path.join(self.model_path, 'all_labels.txt')
        with open(labels_file, 'r') as f:
            self.all_labels = [line.strip() for line in f]
            
        # Separate Intents and Tags (Tags start with 'tag_')
        self.tag_labels = [l for l in self.all_labels if l.startswith('tag_')]
        self.intent_labels = [l for l in self.all_labels if not l.startswith('tag_')]
        
    def predict(self, text: str, tag_threshold=0.5):
        """
        Processes text, runs inference, and returns classified Intent and Tags.
        """
        if not text:
            return "No input text provided.", []

        # 1. Tokenization and Inference
        inputs = self.tokenizer(
            text, 
            padding=True, 
            truncation=True, 
            max_length=self.max_length, 
            return_tensors="pt"
        )
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        logits = outputs.logits.squeeze().cpu().numpy()
        
        # 2. Post-Processing
        
        # A. Intent Classification (First N logits)
        intent_logits = logits[:len(self.intent_labels)]
        predicted_intent_index = np.argmax(intent_logits)
        predicted_intent = self.intent_labels[predicted_intent_index]

        # B. Tag Classification (Remaining M logits)
        tag_logits = logits[len(self.intent_labels):]
        tag_probabilities = 1 / (1 + np.exp(-tag_logits)) # Sigmoid
        
        predicted_tags = []
        for i, prob in enumerate(tag_probabilities):
            if prob >= tag_threshold:
                # Store the tag name, cleaned of the 'tag_' prefix
                predicted_tags.append(self.tag_labels[i].replace('tag_', ''))

        return predicted_intent, predicted_tags