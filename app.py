import gradio as gr
import tensorflow as tf
from PIL import Image
import numpy as np

model_path = "sentinel_classifier_model.keras"
model = tf.keras.models.load_model(model_path)


labels = ['AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 'Industrial', 'Pasture', 'PermanentCrop', 'Residential', 'River', 'SeaLake']
    
def predict_image(image):

    image = Image.fromarray(image.astype('uint8')) 
    image = image.resize((128, 128)) 
    image = np.array(image) / 255.0  

  
    if image.ndim == 2:
        image = np.stack((image,)*3, axis=-1)

    prediction = model.predict(image[None, ...])
    confidences = {labels[i]: float(prediction[0][i]) for i in range(len(labels))}
    return confidences


input_image = gr.Image()
output_text = gr.Textbox(label="Predicted Value")


iface = gr.Interface(
    fn=predict_image,
    inputs=input_image, 
    outputs=gr.Label(),
    title="Sentinel Classifier",
    examples=["images/forest.jpg", "images/highway.jpg", "images/industrial.jpg", "images/residential.jpg", "images/river.jpg"],
    description="Upload a satellite image and the classifier will predict what it is."
)



iface.launch()
