import cv2
import numpy as np
from PIL import Image
import streamlit as st

def preprocess_image(image, target_size=(64, 64)):
    """
    Preprocess image for model prediction
    
    Args:
        image: PIL Image or numpy array
        target_size: tuple, target size for resizing
    
    Returns:
        preprocessed_image: numpy array ready for model input
    """
    try:
        # Convert PIL image to numpy array if needed
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # Convert RGB to BGR if needed (OpenCV uses BGR)
        if len(image.shape) == 3 and image.shape[2] == 3:
            # Check if it's RGB (PIL) or BGR (OpenCV)
            # For consistency, we'll work with RGB
            if image.dtype == np.uint8 and np.max(image) <= 255:
                pass  # Already in correct format
        
        # Resize image
        image_resized = cv2.resize(image, target_size)
        
        # Normalize pixel values to [0, 1]
        image_normalized = image_resized.astype(np.float32) / 255.0
        
        # Add batch dimension
        image_batch = np.expand_dims(image_normalized, axis=0)
        
        return image_batch
        
    except Exception as e:
        st.error(f"Error in image preprocessing: {str(e)}")
        return None

def load_and_preprocess_image(image_file, target_size=(64, 64)):
    """
    Load image from file and preprocess it
    
    Args:
        image_file: uploaded file object from Streamlit
        target_size: tuple, target size for resizing
    
    Returns:
        preprocessed_image: numpy array ready for model input
    """
    try:
        # Open image
        image = Image.open(image_file)
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Preprocess
        preprocessed = preprocess_image(image, target_size)
        
        return preprocessed
        
    except Exception as e:
        st.error(f"Error loading and preprocessing image: {str(e)}")
        return None

def preprocess_for_efficientnet(image):
    """
    Preprocess image specifically for EfficientNet model
    
    Args:
        image: PIL Image or numpy array
    
    Returns:
        preprocessed_image: numpy array ready for EfficientNet input
    """
    return preprocess_image(image, target_size=(224, 224))

def display_image_info(image):
    """
    Display information about the uploaded image
    
    Args:
        image: PIL Image
    """
    if isinstance(image, Image.Image):
        st.write(f"**Image Size:** {image.size}")
        st.write(f"**Image Mode:** {image.mode}")
        st.write(f"**Image Format:** {image.format}")
    elif isinstance(image, np.ndarray):
        st.write(f"**Image Shape:** {image.shape}")
        st.write(f"**Image Data Type:** {image.dtype}")

def resize_image_for_display(image, max_width=500):
    """
    Resize image for better display in Streamlit
    
    Args:
        image: PIL Image
        max_width: maximum width for display
    
    Returns:
        resized_image: PIL Image
    """
    if isinstance(image, Image.Image):
        width, height = image.size
        if width > max_width:
            new_height = int((height * max_width) / width)
            image = image.resize((max_width, new_height), Image.Resampling.LANCZOS)
    
    return image