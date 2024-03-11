# Standard Library
from pathlib import Path
import time

# Third Party
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Streamlit
import streamlit as st

# EgoHOS
from modules import Pipeline, Depth

@st.cache_resource()
def init_models():
    depth = Depth()
    pipe = Pipeline(work_dirs='./work_dirs')
    return depth, pipe

depth, pipe = init_models()

def rgb_to_pointclouds(image: np.ndarray) -> plt.Figure:
    global depth, pipe
    # depth estimate
    st = time.time()
    depth_map = depth.infer(image)
    depth_time = time.time() - st

    # segment
    st = time.time()
    seg_mask = pipe(image)
    hand_mask, obj_mask = seg_mask['hands'], seg_mask['objects']
    seg_time = time.time() - st

    # sample
    st = time.time()
    pcds = depth.sample_pointcloud_open3d(
        depth=depth_map,
        masks=[hand_mask, obj_mask]
    )
    sample_time = time.time() - st

    # show
    pcd_fig = depth.visualize_pointcloud(
        size=(image.shape[1], image.shape[0]),
        pcds=[pcds[0],pcds[1]],
        rgbs=[(244,194,155),(83,109,254)],
        s=0.5,
        out='plt'
    )

    return pcd_fig, depth_time, seg_time, sample_time



st.title('RGB to Hand-Object PointClouds')

choice = st.radio(
    "Select an image source:",
    ["**From Files**", ":rainbow[From Camera]"],
    horizontal=True
)

if choice == ":rainbow[From Camera]":
    image = st.camera_input("Take an image from webcam")
elif choice == "**From Files**":
    image = st.file_uploader("Upload an image", type=['jpg', 'png', 'jpeg'])

if image is not None:
    img = Image.open(image)
    img_array = np.array(img)
    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    fig, t1, t2, t3 = rgb_to_pointclouds(img_array)

    # plot
    st.pyplot(fig)
    # write time
    st.write(f"Depth Time: {t1:.2f} sec")
    st.write(f"Segment Time: {t2:.2f} sec")
    st.write(f"Sample Time: {t3:.2f} sec")