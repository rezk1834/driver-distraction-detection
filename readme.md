# 🚗 Driver Distraction Detection System

A deep learning-powered web application that detects and classifies driver distraction behaviors using computer vision. Built with Streamlit and TensorFlow.

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-v2.13.0-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-v1.28.1-red.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## 📋 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Demo](#demo)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [Deployment](#deployment)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

This system uses advanced deep learning models to automatically detect and classify driver distraction behaviors from images. It can identify 10 different types of driver activities, helping improve road safety through automated monitoring.

### Classification Categories:
- ✅ Safe driving
- 📱 Texting (right/left hand)
- ☎️ Talking on phone (right/left hand)
- 📻 Operating the radio
- 🥤 Drinking
- 🔄 Reaching behind
- 💄 Hair and makeup
- 👥 Talking to passenger

## ✨ Features

- **Real-time Detection**: Upload images for instant distraction detection
- **Multiple Models**: Choose between Custom CNN and EfficientNetB3
- **Interactive UI**: User-friendly Streamlit interface
- **Confidence Scoring**: Get prediction confidence levels
- **Visual Feedback**: Color-coded results based on safety levels
- **Model Comparison**: Compare performance between different models

## 🚀 Demo

You can try the live demo [here](your-deployment-url) or run it locally following the installation instructions.

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- pip

### Local Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/driver-distraction-detection.git
   cd driver-distraction-detection
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download model files**
   - Place your trained model files in the `models/` directory:
     - `driver_model2.keras` (Custom CNN model)
     - `efficientnetb3.keras` (EfficientNetB3 model)

5. **Run the application**
   ```bash
   streamlit run app.py
   ```

The app will be available at `http://localhost:8501`

## 📖 Usage

1. **Launch the application** using the command above
2. **Select a model** from the dropdown (Custom CNN or Efficient