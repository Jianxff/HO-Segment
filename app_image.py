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

@st.cache_resource()
def init_models():
    # EgoHOS
    from modules import HOSegment, Depth
    depth = Depth(absolute=False)
    pipe = HOSegment(work_dirs='./work_dirs', parallel=True)
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
    from modules import HOSegment, Depth
    seg_pic = HOSegment.visualize(image, seg_mask)
    seg_pic = cv2.cvtColor(seg_pic, cv2.COLOR_BGR2RGB)
    depth_pic = Depth.gray(depth_map, bgr=True)

    return pcd_fig, seg_pic, depth_pic, depth_time, seg_time, sample_time



st.title('RGB to Hand-Object PointClouds')


choice = st.radio(
    "Select an image source:",
    ["**From Files**", ":rainbow[From Camera]"],
    horizontal=True
)

main_col, plot_col = st.columns(2)

if choice == ":rainbow[From Camera]":
    image = main_col.camera_input("Take an image from webcam")
elif choice == "**From Files**":
    image = main_col.file_uploader("Upload an image", type=['jpg', 'png', 'jpeg'])

if image is not None:
    img = Image.open(image)
    img_array = np.array(img)
    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    fig, pic, depth, t1, t2, t3 = rgb_to_pointclouds(img_array)

    # cols
    col1, col2 = main_col.columns(2)
    col1.image(depth, caption=f'Depth Estimation in {t1:.2f}s')
    col2.image(pic, caption=f'Segmentation in {t2:.2f}s')

    # pintcloud
    plot_col.write(f"Sample PointCloud in {t3:.2f}s")
    # pyplot
    elev = plot_col.slider("elev", min_value=0, max_value=90, value=45)
    azim = plot_col.slider("azim", min_value=0, max_value=360, value=180)
    fig_, ax = fig
    ax.view_init(elev=elev, azim=azim)
    plot_col.pyplot(fig_)
    