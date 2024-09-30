from PIL import Image
import streamlit as st
import pandas as pd


from transformers import pipeline

pipe = pipeline("image-classification", model="julien-c/hotdog-not-hotdog")


uploaded_files = st.file_uploader(
    "Choose a file", accept_multiple_files=True
)
for uploaded_file in uploaded_files:
    imag = Image.open(uploaded_file)    
    pred = pipe(imag)
    st.write("filename:", uploaded_file.name)
    st.write(pred)
