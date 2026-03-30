import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
# import tensorflow as tf
from tensorflow.keras.models import load_model


class_names = ["CBFB_MYH11", "NPM1", "Normal", "PML_RARA", "RUNX1_RUNX1T1"]


@st.cache_resource
def get_model():
    model = load_model("model/model.keras")
    return model


model = get_model()


def sample_predict(model, image):
    img = image.resize((144, 144))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)

    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions[0])
    predicted_class = class_names[predicted_class_index]
    confidence = float(predictions[0][predicted_class_index])

    return predicted_class, confidence


st.title("Leukemia Cell Classification")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image = image.convert("RGB")
    st.image(image, caption="Uploaded Image", width=400)

    if st.button("Predict"):
        predicted_class, confidence = sample_predict(model, image)

        st.write(f"## Predicted class: *{predicted_class}*")
        st.write(f"## Confidence: {confidence:.2f}")
