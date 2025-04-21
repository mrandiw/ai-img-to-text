![PythonAnywhere](https://img.shields.io/badge/pythonanywhere-%232F9FD7.svg?style=for-the-badge&logo=pythonanywhere&logoColor=151515)
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)

# AI Image-to-Text Converter

This project converts images to descriptive text using AI models, leveraging computer vision and natural language processing capabilities through the Transformers library.

## Project Overview

This tool allows users to upload images and receive AI-generated text descriptions of the content. It uses pre-trained vision-language models to analyze images and produce accurate textual descriptions.

## Features

- Image-to-text conversion using state-of-the-art AI models
- User-friendly interface through Gradio
- Support for various image formats
- GPU acceleration for faster processing

## Supported Models

### Salesforce/blip-image-captioning-base

This project uses the BLIP (Bootstrapping Language-Image Pre-training) model from Salesforce for image captioning. The base model offers a good balance between performance and resource requirements.

Key features:
- Pre-trained on over 14 million image-text pairs
- Effective for generating natural language descriptions of images
- Supports zero-shot image captioning

Example usage in code:
```python
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

# Load model and processor
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Process image
image = Image.open("path/to/image.jpg")
inputs = processor(image, return_tensors="pt")

# Generate caption
output = model.generate(**inputs)
caption = processor.decode(output[0], skip_special_tokens=True)
print(caption)
```

## Setup Instructions

### 1. Environment Preparation
Create and activate a Python virtual environment:
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# macOS/Linux
python -m venv venv
source venv/bin/activate

# Ubuntu
python -m venv venv
source venv/bin/activate
```

### 2. Install Required Libraries
Install the necessary Python packages:
```bash
pip install pillow
pip install transformers
pip install gradio
# for NVIDIA GPU
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 
```

## Usage

After installation, run the application using:
```bash
python main.py
```

Then open your web browser and navigate to the URL displayed in the terminal (typically http://127.0.0.1:7860).

## Requirements

- Python 3.8 or higher
- CUDA-compatible GPU (optional, for faster processing)
- Internet connection (for initial model download)