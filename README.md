-----

## 🛠️ Setup and Installation

Follow these steps to set up the environment, install dependencies, and prepare the model.

### 1\. Prerequisites

  * Python 3.8+
  * The raw data file: `Bitext_Sample_Customer_Service_Training_Dataset.csv` must be placed in the project root directory.

### 2\. Environment Setup

It is highly recommended to use a virtual environment (`venv`) to isolate project dependencies:

```bash
# 1. Navigate to the project root
cd customer_intent_tagger

# 2. Create the virtual environment
python -m venv .venv

# 3. Activate the environment
# On Windows
.\.venv\Scripts\activate

# On macOS/Linux
source .venv/bin/activate
```

### 3\. Install Dependencies

Install all required Python packages using the provided `requirements.txt` file:

```bash
pip install -r requirements.txt
```

-----

## 🏃 Usage and Execution

### 1\. Train the Model

The `model_trainer.py` script executes the full training pipeline: data loading, preprocessing, train/validation split, model fine-tuning (DistilBERT), and saving the final model.

Run the training script from the project root:

```bash
# Ensure your .venv is active
python src/model_trainer.py
```

  * **Note:** This process will take several minutes. Once complete, the model will be saved to `models/bert_classifier/final_model/`.

### 2\. Run the Web Application

Once the model is trained and saved, you can launch the interactive Streamlit interface:

```bash
streamlit run app.py
```

The application will open in your web browser, allowing you to enter customer utterances and see the real-time Intent and Tag classifications.

-----

## 🧠 Model Details

| Feature | Detail |
| :--- | :--- |
| **Base Model** | DistilBERT (`distilbert-base-uncased`) |
| **Task Type** | Multi-Task Classification |
| **Heads** | **Intent** (Multi-Class) + **Tags** (Multi-Label) |
| **Loss Function** | Cross-Entropy (handled internally by Hugging Face `Trainer`) |
| **Target Labels** | 12 Intents (e.g., `cancel_order`) + 26 Tags (e.g., `tag_B`, `tag_I`, `tag_P`) |
| **Final Metrics** | Evaluated using **F1-Macro** (best model selection), Accuracy, and ROC AUC. |

-----

## 🤝 Contribution and Issues

If you find any issues or have suggestions, please open an issue in the repository.

-----
