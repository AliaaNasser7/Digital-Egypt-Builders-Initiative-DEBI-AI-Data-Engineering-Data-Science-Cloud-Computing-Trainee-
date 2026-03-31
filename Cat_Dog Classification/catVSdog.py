import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import numpy as np
model = load_model("cat_dog_model.h5")

st.title("Cat vs Dog Classifier")

uploaded_file = st.file_uploader("Upload an image", type=["jpg","png","jpeg"])

if uploaded_file is not None:
    
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_container_width=True)
    img = img.resize((128,128))
    img_array = image.img_to_array(img)
    img_array = img_array/255
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    prob = prediction[0][0]

    if prob > 0.5:
        st.success(f"Prediction: Dog ({prob*100:.2f}%)")
    else:
        st.success(f"Prediction: Cat ({(1-prob)*100:.2f}%)")
        
        