import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from streamlit_drawable_canvas import st_canvas
from PIL import Image

model = load_model("mnist_model.h5")

st.title("Digit Recognition")

canvas_result = st_canvas(
    stroke_width=15,
    stroke_color="white",
    background_color="black",
    height=280,
    width=280,
    drawing_mode="freedraw",
)

if canvas_result.image_data is not None:

    img = canvas_result.image_data

    img = Image.fromarray(img.astype('uint8'))
    img = img.convert("L")
    img = img.resize((28,28))

    img = np.array(img) / 255.0
    img = img.reshape(1,28,28,1)

    prediction = model.predict(img)

    digit = np.argmax(prediction)

    st.subheader(f"Prediction: {digit}")