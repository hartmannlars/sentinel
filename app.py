import gradio as gr
import tensorflow as tf
from PIL import Image
import numpy as np

# Load the model
model_path = "sentinel_classification_model.keras"
model = tf.keras.models.load_model(model_path)

# Define labels
labels = ['AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 'Industrial', 'Pasture', 'PermanentCrop', 'Residential', 'River', 'SeaLake']

def predict_image(image):
    image = Image.fromarray(image.astype('uint8'), 'RGB')
    image = image.resize((64, 64)) 
    image = np.array(image)

    prediction = model.predict(np.expand_dims(image, axis=0))
    confidences = {labels[i]: float(prediction[0][i]) for i in range(len(labels))}
    return confidences

# Gradio interface
iface = gr.Interface(
    fn=predict_image,
    inputs=gr.Image(),
    outputs=gr.Label(num_top_classes=10),
    title="Sentinel Image Classifier",
    description="Upload a satellite image and the classifier will predict the type of land cover or feature.",
    examples=["images/forest.jpg", "images/highway.jpg", "images/industrial.jpg", "images/residential.jpg", "images/river.jpg"]
)

iface.launch()
