# Standard Library
from pathlib import Path
import argparse

# Third Party
import cv2
import numpy as np

# Streamlit
import streamlit as st
from streamlit_webrtc import webrtc_streamer
import av

# EgoHOS
from modules import Pipeline, Streaming

@st.cache_resource()
def create_pipe():
    return Pipeline(work_dirs='./work_dirs')

pipe = create_pipe()

@st.cache_resource()
def create_streaming():
    return Streaming(pipe.infer, limit=240)


streaming = create_streaming()

def video_frame_callback(frame: np.ndarray):
    """
    Callback function to be used with webrtc_streamer
    """
    img = frame.to_ndarray(format="bgr24")

    streaming.push(img)
    data = streaming.get()

    if data is not None:
        img = Pipeline.visualize(image=img, result=data).astype(np.uint8)

    return av.VideoFrame.from_ndarray(img, format="bgr24")


# streamlit
st.title('Streaming Hand-Object Segmentation')

# webrtc
webrtc_streamer(key="sample", video_frame_callback=video_frame_callback)