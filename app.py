import pickle
import streamlit as st
from PIL import Image, ImageOps
from classifier import image_classification
import matplotlib.pyplot as plt
import numpy as np


# loading the trained model
st.set_option('deprecation.showfileUploaderEncoding', False)
html_temp = """
    <div style ="background-image: url("https://www.jigsawstore.com.au/assets/full/RB15633-7.jpg?20200726170018"); background-repeat: no-repeat; background-attachment: fixed; background-size: 100% 100%;} body::before{content: ""; position: absolute; top: 0px; right: 0px; bottom: 0px; left: 0px; background-color: rgba(1,2,1,0.80);">
    <div style ="background-color:tomato;padding:13px"> 
    <h1 style ="font-family:verdana;color:white;text-align:center;">DOG BREED PREDICTOR</h1> 
    </div> 
    """
# display the front end aspect
st.markdown(html_temp, unsafe_allow_html = True) 
# st.markdown('<style>body{background-image: url("https://www.jigsawstore.com.au/assets/full/RB15633-7.jpg?20200726170018"); background-repeat: no-repeat; background-attachment: fixed; background-size: 100% 100%;} body::before{content: ""; position: absolute; top: 0px; right: 0px; bottom: 0px; left: 0px; background-color: rgba(1,2,1,0.80);}</style>',unsafe_allow_html=True)
# st.markdown('<style>body{color: tomato; text-align: center;}</style>',unsafe_allow_html=True)
# st.markdown()


st.header("Upload an image of a dog to identify it's breed :dog:")

st.write("")
st.write("")
st.write("")
st.write("")
st.subheader("Choose a dog image... :dog:")

# AND in st.sidebar!
with st.sidebar:
      if st.button("About"):
          st.write("Visit [Github](https://github.com/celestial-shubham/Dog-Breed-Predictor) !!")
          st.text("By Shubham Verma")
            
uploaded_file = st.file_uploader("", type=["jpg"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded image.', use_column_width=True)
    st.write("")
    with st.spinner('Identifying...'):
        label = image_classification(image,"/content/drive/MyDrive/Dog breed prediction/Model/20201214-13301607952655-full-image-set-mobilenetv2-Adam.h5")

    btn = st.button("See Results!!")
    if btn :
      st.info(label)
      st.balloons()
