"""
Configuration file for the Driver Distraction Detection app
"""

# Class labels mapping
CLASS_LABELS = {
    0: 'Safe driving',
    1: 'Texting - right',
    2: 'Talking on the phone - right',
    3: 'Texting - left',
    4: 'Talking on the phone - left',
    5: 'Operating the radio',
    6: 'Drinking',
    7: 'Reaching behind',
    8: 'Hair and makeup',
    9: 'Talking to passenger'
}

# Model configurations
MODEL_CONFIGS = {
    'custom_cnn': {
        'input_size': (64, 64, 3),
        'file_name': 'driver_model2.keras',
        'description': 'Custom CNN with 4 convolutional layers'
    },
    'efficientnetb3': {
        'input_size': (224, 224, 3),
        'file_name': 'efficientnetb3.keras',
        'description': 'EfficientNetB3 transfer learning model'
    }
}

# Image processing parameters
IMG_SIZE = {
    'custom_cnn': (64, 64),
    'efficientnetb3': (224, 224)
}

# Model URLs (if hosting models externally)
MODEL_URLS = {
    'custom_cnn': 'https://your-storage-url.com/driver_model2.keras',
    'efficientnetb3': 'https://your-storage-url.com/efficientnetb3.keras'
}

# App settings
APP_SETTINGS = {
    'max_file_size': 10,  # MB
    'allowed_extensions': ['png', 'jpg', 'jpeg'],
    'default_model': 'efficientnetb3'
}

# Safety thresholds
SAFETY_THRESHOLDS = {
    'high_confidence': 0.8,
    'medium_confidence': 0.6,
    'low_confidence': 0.4
}

# Color schemes for predictions
PREDICTION_COLORS = {
    0: '#4CAF50',  # Safe - Green
    1: '#F44336',  # Texting right - Red
    2: '#FF5722',  # Phone right - Red Orange
    3: '#F44336',  # Texting left - Red
    4: '#FF5722',  # Phone left - Red Orange
    5: '#FF9800',  # Radio - Orange
    6: '#FFC107',  # Drinking - Amber
    7: '#FF9800',  # Reaching - Orange
    8: '#FFC107',  # Hair/makeup - Amber
    9: '#FF9800'   # Talking - Orange
}

# Sample images for demonstration (you would replace these with actual image paths)
SAMPLE_IMAGES = {
    'safe_driving': 'assets/images/sample_images/safe_driving.jpg',
    'texting': 'assets/images/sample_images/texting.jpg',
    'phone_call': 'assets/images/sample_images/phone_call.jpg',
    'drinking': 'assets/images/sample_images/drinking.jpg',
    'radio': 'assets/images/sample_images/radio.jpg'
}