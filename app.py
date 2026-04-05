import numpy as np
from PIL import Image
import gradio as gr
from tensorflow.keras.models import load_model

# Load model
model = load_model("model.h5")

def predict_digit(image):
    # Convert to grayscale
    img = image.convert('L')

    # Resize to 28x28
    img = img.resize((28, 28))

    # Normalize
    input_data = np.array(img) / 255.0

    # Reshape
    input_data = input_data.reshape(1, 28, 28)

    # Prediction
    prediction = model.predict(input_data)
    digit = np.argmax(prediction)

    return f"Predicted Digit: {digit}"

# Gradio UI
interface = gr.Interface(
    fn=predict_digit,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="Handwritten Digit Recognizer"
)

interface.launch()