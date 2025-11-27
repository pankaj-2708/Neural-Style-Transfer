import streamlit as st
import time
from util import *


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
        
    with col2:
        placeholder_predict = st.empty()
        
        predict = st.button("Generate Output of given images")
        # print(predict)
        
    if predict:
        try:
            id=upload_images(style_image,content_image)
        except Exception as E:
            print(E)
            st.error("An unexpected error occured")
        try:
            placeholder_predict.empty()
            output_image=predict_output(id)
        except Exception as E:
            st.write(E)    

        with col2:
            st.markdown("\n")
            st.markdown(
            "<h2 style='text-align: center;'>Output Image</h2>",
            unsafe_allow_html=True,
        )
            
            st.image(output_image,caption="Output Image")
            
            
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
    
    id=upload_video(style_image,content_video)
    output=predict_output_video(id)
      
    with col2:
        st.markdown("\n")
        st.markdown(
        "<h2 style='text-align: center;'>Output Video</h2>",
        unsafe_allow_html=True,
    )
        st.video(output, format="video/mp4")