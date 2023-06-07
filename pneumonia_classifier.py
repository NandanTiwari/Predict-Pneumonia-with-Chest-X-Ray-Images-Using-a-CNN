import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array

import streamlit as st
import sys
import pandas as pd
import os


# In[2]:
st.set_option('deprecation.showPyplotGlobalUse', False)
st.enableXsrfProtection = True
# st.markdown(
#     """
#     <style>
#       .main{
#       background-color:#043B4D ;
#       }
#       </style>

#     """,
#     unsafe_allow_html=True
# ) 

from datetime import date
import time
model = tf.keras.models.load_model("Resnet_model.h5")

st.title("Pneumonia Prediction")
st.image("https://i.imgur.com/jZqpV51.png",use_column_width=True)
st.subheader("Developed by Nandan Tiwari")
today = date.today()
today=today.strftime("%d/%m/%Y")
st.write("TODAY:-",today)
st.sidebar.title("About Pneumonia")
st.sidebar.write("Pneumonia is an infection that inflames the air sacs in one or both lungs. "
                 "It can range in severity from mild to life-threatening. Chest X-ray imaging "
                 "is commonly used to diagnose pneumonia and assess its extent.")

st.write("This is a pneumonia prediction project that uses a deep learning model to classify chest X-ray images.")

uploaded_file = st.file_uploader("Upload a chest X-ray image", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = load_img(uploaded_file, target_size=(150, 150))
    image_array = img_to_array(image)
    image_array /= 255.0
    image_array = np.expand_dims(image_array, axis=0)
    predictions = model.predict(image_array)
    confidence = np.max(predictions)
    st.write(f"Prediction: {predictions}")
    with st.spinner('Analyzing.......'):
        time.sleep(0.2)
        st.success('Done!')
#     st.write(predictions)
#     st.write("with",confidence*100,"% confidence")
    threshold = 0.6
    st.image(image, use_column_width=True)
    st.subheader("Prediction:")
    if confidence<threshold:
        st.write("Prediction: Normal")
    else:
        st.write("Prediction: Pneumonia")
 
    st.write(f"Model Confidence:{confidence:.2f}")

