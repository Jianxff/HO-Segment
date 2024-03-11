# Standard Library
from pathlib import Path
import argparse
import time

# Third Party
import cv2
from tqdm import tqdm
import numpy as np

# HO Segment
from modules import Pipeline, Depth

depth = Depth()
pipe = Pipeline(work_dirs='./work_dirs')

def predict(frame: np.ndarray):
    # depth
    depth_map = depth.infer(frame)

    if depth_map is None:
        return None

    st = time.time()
    # segment
    seg_mask = pipe(frame)
    hand_mask, obj_mask = seg_mask['hands'], seg_mask['objects']
    nd = time.time()
    print(f"Segment Time: {nd-st:.2f} sec")
    st = nd

    # sample
    # pcds = depth.sample(
    #     depth=depth_map,
    #     masks=[hand_mask, obj_mask]
    # )

    # nd = time.time()
    # print(f"Sample Time: {nd-st:.2f} sec")
    # st = nd

    # pcds = depth.filter(pcds)

    # nd = time.time()
    # print(f"Filter Time: {nd-st:.2f} sec")
    pcds = depth.sample_pointcloud_open3d(
        depth=depth_map,
        masks=[hand_mask, obj_mask]
    )
    nd = time.time()
    print(f'Filter Sample Time: {nd-st:.2f} sec')
    st = nd
    
    # show
    seg_view = pipe.visualize(frame, seg_mask)
    pcd_view = depth.visualize_pointcloud(
        size=(frame.shape[1], frame.shape[0]),
        # pcds=[
        #     depth.downsample_pointcloud(pcds[0], -50), 
        #     depth.downsample_pointcloud(pcds[1], -50),
        # ],
        pcds=[pcds[0],pcds[1]],
        rgbs=[
            (244,194,155),
            (83,109,254)
        ],
        s=0.5
    )


    depth_map_gray = depth.gray(depth_map, bgr=True)
    depth_view = cv2.resize(depth_map_gray,dsize=None,fx=0.3,fy=0.3)
    seg_view[0:depth_view.shape[0], 0:depth_view.shape[1]] = depth_view

    view = cv2.hconcat([seg_view, pcd_view])

    nd = time.time()
    print(f"Visualize Time: {nd-st:.2f} sec")

    return view

def predict_image(args):
    frame = cv2.imread(args.image)
    view = predict(frame)
    cv2.imwrite(args.save, view)

def predict_video(args):
    # Load video
    cap = cv2.VideoCapture(args.video)

    if not cap.isOpened():
        raise ValueError("Video file could not be opened")

    # width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    # height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = None

    for _ in tqdm(range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))):  
        ret, frame = cap.read()
        if not ret:
            break
        view = predict(frame)
        if view is None: continue

        if not out:
            out = cv2.VideoWriter(
                args.save,
                cv2.VideoWriter_fourcc(*'mp4v'),
                30,
                (view.shape[1], view.shape[0])
            )
        
        out.write(view)

    cap.release()
    out.release()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Predict depth from a video file")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--image", type=str, help="Path to input image file")
    group.add_argument("--video", type=str, help="Path to input video file")
    parser.add_argument("--save", type=str, default='result')

    args = parser.parse_args()

    if args.image and args.video:
        raise ValueError("Please specify either --image or --video")

    if args.image:
        predict_image(args)
    elif args.video:
        predict_video(args)


