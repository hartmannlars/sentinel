# Satellite Image Classification Project

This repository contains all resources and code for the Satellite Image Classification project conducted as a study project at the ZHAW School of Management and Law. The project uses machine learning to classify satellite images from the EuroSAT dataset, which are derived from Sentinel satellites, into ten different land cover types.

## Project Overview

The aim of this project is to develop an automated classification system to help with environmental monitoring and urban planning by classifying satellite images into categories such as Forest, Highway, Residential, and others. It utilizes TensorFlow for model development and Gradio for deployment to provide an interactive web interface.

## Features

- **Automated Image Classification:** Classify satellite images into ten distinct land cover categories.
- **Interactive Web Application:** A Gradio-based interface that allows users to upload satellite images and view the classification results in real time.
- **Data Augmentation Techniques:** Employ various augmentation techniques to improve the model's accuracy and robustness.

## Installation

To set up this project locally, follow these steps:

```bash
git clone https://github.com/hartmannlars/satellite-image-classification.git
cd satellite-image-classification
pip install -r requirements.txt
```

## Usage

Launch the application with the following command:

```bash
python app.py
```

Then, navigate to http://localhost:7860 in your web browser to interact with the model.

## Usage

The project uses the EuroSAT dataset, which comprises 30,988 labeled satellite images categorized into 10 classes.

## Model Architecture

The model is based on the EfficientNetB0 architecture, utilizing TensorFlow for training and fine-tuning on the satellite images.
