import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import streamlit as st
import os
import requests
from io import BytesIO
from utils.image_processing import preprocess_image, preprocess_for_efficientnet

# Cache models to avoid reloading
@st.cache_resource
def load_models(model_type):
    """
    Load the specified model
    
    Args:
        model_type: str, either 'custom_cnn' or 'efficientnetb3'
    
    Returns:
        model: loaded Keras model
    """
    try:
        model_paths = {
            'custom_cnn': 'models/driver_model2.keras',
            'efficientnetb3': 'models/efficientnetb3.keras'
        }
        
        model_path = model_paths.get(model_type)
        
        if model_path and os.path.exists(model_path):
            model = load_model(model_path)
            return model
        else:
            # If local model doesn't exist, you might want to download from a URL
            # or create a placeholder model for demonstration
            st.warning(f"Model file {model_path} not found. Using placeholder model.")
            return create_placeholder_model(model_type)
            
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def create_placeholder_model(model_type):
    """
    Create a placeholder model for demonstration when actual model files are not available
    
    Args:
        model_type: str, type of model to create
    
    Returns:
        model: simple Keras model for demonstration
    """
    try:
        if model_type == 'custom_cnn':
            input_shape = (64, 64, 3)
        else:
            input_shape = (224, 224, 3)
        
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=input_shape),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(10, activation='softmax')
        ])
        
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        
        st.info("Using placeholder model for demonstration. Replace with your trained model files.")
        return model
        
    except Exception as e:
        st.error(f"Error creating placeholder model: {str(e)}")
        return None

def predict_with_model(image, model, model_type):
    """
    Make prediction using the specified model
    
    Args:
        image: PIL Image
        model: loaded Keras model
        model_type: str, type of model being used
    
    Returns:
        prediction: int, predicted class index
        confidence: float, confidence score of the prediction
    """
    try:
        # Preprocess image based on model type
        if 'efficientnet' in model_type.lower():
            processed_image = preprocess_for_efficientnet(image)
        else:
            processed_image = preprocess_image(image, target_size=(64, 64))
        
        if processed_image is None:
            raise ValueError("Image preprocessing failed")
        
        # Make prediction
        predictions = model.predict(processed_image, verbose=0)
        
        # Get the predicted class and confidence
        predicted_class = np.argmax(predictions[0])
        confidence = np.max(predictions[0])
        
        return predicted_class, confidence
        
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        return 0, 0.0  # Return safe driving as default

def get_prediction_probabilities(image, model, model_type):
    """
    Get full prediction probabilities for all classes
    
    Args:
        image: PIL Image
        model: loaded Keras model
        model_type: str, type of model being used
    
    Returns:
        probabilities: numpy array of probabilities for each class
    """
    try:
        # Preprocess image based on model type
        if 'efficientnet' in model_type.lower():
            processed_image = preprocess_for_efficientnet(image)
        else:
            processed_image = preprocess_image(image, target_size=(64, 64))
        
        if processed_image is None:
            raise ValueError("Image preprocessing failed")
        
        # Make prediction
        predictions = model.predict(processed_image, verbose=0)
        
        return predictions[0]
        
    except Exception as e:
        st.error(f"Error getting prediction probabilities: {str(e)}")
        return np.zeros(10)  # Return zeros for all classes

def download_model_from_url(url, local_path):
    """
    Download model from URL and save locally
    
    Args:
        url: str, URL to download model from
        local_path: str, local path to save the model
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        st.info(f"Downloading model from {url}...")
        
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        
        with open(local_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        st.success(f"Model downloaded successfully to {local_path}")
        return True
        
    except Exception as e:
        st.error(f"Error downloading model: {str(e)}")
        return False

def validate_model_input(image, model_type):
    """
    Validate that the image is suitable for the model
    
    Args:
        image: PIL Image
        model_type: str, type of model
    
    Returns:
        bool: True if valid, False otherwise
    """
    try:
        if image is None:
            return False
        
        # Check image format
        if not hasattr(image, 'size'):
            return False
        
        # Check image is not empty
        if image.size[0] == 0 or image.size[1] == 0:
            return False
        
        return True
        
    except Exception:
        return False