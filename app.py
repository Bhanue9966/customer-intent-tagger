# app.py

import streamlit as st
import os
import sys

# Add the 'src' directory to the path so we can import modules from it
sys.path.append(os.path.abspath('src'))

# Use st.cache_resource to load the model only once
@st.cache_resource
def get_predictor():
    from src.predictor import IntentTaggerPredictor

    # Define the expected path to the saved model relative to the project root
    MODEL_PATH = 'models/bert_classifier/final_model' 
    
    # Check if the model directory exists
    if not os.path.isdir(MODEL_PATH):
        st.error(f"Model directory not found at: {MODEL_PATH}")
        st.error("Please run 'python src/model_trainer.py' first to train and save the model.")
        return None

    # Load the predictor
    return IntentTaggerPredictor(MODEL_PATH)

# --- Streamlit App Layout ---

st.set_page_config(page_title="Customer Intent & Tag Classifier", layout="wide")

st.title("🗣️ Customer Service Utterance Classifier")
st.markdown("This application uses a fine-tuned DistilBERT model to classify customer input into a specific **Intent** and multiple **Tags**.")

# Load the model predictor
predictor = get_predictor()

if predictor:
    
    st.header("1. Input Customer Utterance")
    
    default_text = "I am having a serious problem with my payment, I need assistance urgently to avoid a late fee."
    user_input = st.text_area(
        "Enter customer text here:",
        default_text,
        height=150
    )
    
    # Threshold slider for the multi-label tags
    tag_threshold = st.slider(
        "Confidence Threshold for Tags (Higher = Stricter)",
        min_value=0.1,
        max_value=0.9,
        value=0.5,
        step=0.05
    )
    
    if st.button("Classify Utterance", type="primary"):
        if user_input:
            
            # Run Prediction
            with st.spinner("Analyzing text and predicting intent/tags..."):
                predicted_intent, predicted_tags = predictor.predict(user_input, tag_threshold=tag_threshold)
            
            st.success("Analysis Complete!")
            
            # --- Output Display ---
            st.header("2. Classification Results")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🎯 Primary Intent (What the customer wants)")
                st.metric(label="Predicted Intent", value=predicted_intent.upper().replace('_', ' '))
                

            with col2:
                st.subheader("🏷️ Style/Emotion Tags (How they are saying it)")
                if predicted_tags:
                    tags_str = ", ".join([tag.upper() for tag in predicted_tags])
                    st.info(tags_str)
                else:
                    st.warning("No tags found above the set threshold.")