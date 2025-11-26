import streamlit as st
# import cv2
from PIL import Image
import io
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import time
from src.Frontend.util import *

st.set_page_config(page_title="Neural Style Transfer",layout="wide")

st.markdown(
        "<h1 style='text-align: center;'>Neural Style Transfer</h1>",
        unsafe_allow_html=True,
    )

st.markdown(
        "<h2 style='text-align: center;'>Image Section</h2>",
        unsafe_allow_html=True,
    )
st.markdown("\n")
st.markdown("\n")

col1,col2=st.columns(2)
with col1:
    st.subheader("Upload Style image here")
    style_image=st.file_uploader(label="StyleI mage",type=["jpg", "jpeg", "png"],key=1)

with col2:
    st.subheader("Upload Content image here")
    content_image=st.file_uploader(label="Content Image",type=["jpg", "jpeg", "png"],key=2)


if style_image and content_image:
    
    col1, col2, col3 = st.columns([1,.5,1])
    id=upload_images(style_image,content_image)
    
    with col2:
        placeholder_loading = st.empty()
        placeholder_predict = st.empty()
        
        predict = st.button("Generate Output of given images")
        # print(predict)
        
    if predict:
        try:
            placeholder_predict.empty()
            placeholder_loading.image("./src/Frontend/assets/loading.gif")
            output_image=predict_output(id)
        except Exception as E:
            st.write(E)    

    # ouput=input_to_ouput(style_image,content_image,"image")
    
    # display output image
    
    # simulating above operations  
    # time.sleep(10)
        placeholder_loading.empty()
        with col2:
            st.markdown("\n")
            st.markdown(
            "<h2 style='text-align: center;'>Output Image</h2>",
            unsafe_allow_html=True,
        )
            # output_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            st.image(output_image,caption="Output Image",width=300)
        
            pil_img = Image.fromarray(output_image)
            buf = io.BytesIO()
            pil_img.save(buf, format="PNG")
            byte_img = buf.getvalue()

            col1_,col2_,col3_=st.columns([1,3,1])
            
            with col2_:
                st.download_button(
                label="Download Image ",
                data=byte_img,
                file_name="output.png",
                mime="image/png"
                )
            
            
st.markdown(
        "<h2 style='text-align: center;'>Video Section</h2>",
        unsafe_allow_html=True,
    )

col1,col2=st.columns(2)
with col1:
    st.subheader("Upload Style image here")
    style_image=st.file_uploader(label="StyleI mage",type=["jpg", "jpeg", "png"],key=3)

with col2:
    st.subheader("Upload Content video here")
    content_video=st.file_uploader(label="Content Video",type=["mp4", "mkv"],key=4)
    
if style_image and content_video:
    
    col1, col2, col3 = st.columns([1,.5,1])
    with col2:
        placeholder = st.empty()
        placeholder.image("./src/Frontend/assets/loading.gif")
    
    # output=input_to_output(style_image,content_video,"video")
        
    placeholder.empty()
    with col2:
        st.markdown("\n")
        st.markdown(
        "<h2 style='text-align: center;'>Output Video</h2>",
        unsafe_allow_html=True,
    )
        st.video(output_image,caption="Output Video",width=300)
    

        col1_,col2_,col3_=st.columns([1,3,1])
        
        with col2_:
            st.download_button(
            label="Download Image ",
            data=byte_img,
            file_name="output.mp4",
            mime="video/mp4"
            )