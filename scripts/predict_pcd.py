# Standard Library
from pathlib import Path
import argparse
import time
import os
from typing import *

# Third Party
import cv2
from tqdm import tqdm
import numpy as np
import open3d

# HO Segment
from modules import HOSegment, Depth

dep_pipe: Depth = None
seg_pipe: HOSegment = None

## modules
def load_depth_model():
    global dep_pipe    
    if dep_pipe is None:
        print('Loading Depth Estimation Model')
        dep_pipe = Depth(absolute=True)
 
def load_segment_model():
    global seg_pipe
    if seg_pipe is None:
        print('Loading Hand-Object Segmentation Model')
        seg_pipe = HOSegment(work_dirs='./work_dirs')     

def read_images(
    directory: Union[str, Path], 
    depth: Optional[bool] = False,
) -> List[np.ndarray]:
    frames = []
    directory = Path(directory)
    
    if directory.is_file():
        return [cv2.imread(str(directory))]
    
    print('reading images')
    for p in tqdm(sorted(directory.iterdir(), key=lambda x: int(x.stem.split('_')[2]))):
    # for p in tqdm(sorted(directory.iterdir(), key=lambda x: int(x.stem))):
        # print(p)
        if p.suffix.lower() in ['.jpg', '.png']:
            frame = cv2.imread(str(p), cv2.IMREAD_ANYDEPTH if depth else cv2.IMREAD_COLOR)
            frames.append(frame)
    return frames

def sample_rgb(video: Union[str, Path]) -> List[np.ndarray]:
    cap = cv2.VideoCapture(str(video))
    frames = []
    print('Sampling RGB frames from video file')

    for _ in tqdm(range(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames

def estimate_depth(frames: List[np.ndarray]) -> List[np.ndarray]:
    load_depth_model()
    depth_maps = []
    print('Estimating Depth')
    for frame in tqdm(frames):
        depth_map = dep_pipe.infer(frame)
        depth_maps.append(depth_map)
    return depth_maps


## predict function
def predict(frame: np.ndarray, depth: np.ndarray):
    st = time.time()
    # segment
    seg_mask = seg_pipe(frame)
    
    # check empty
    if not (seg_mask['hands_t'] and seg_mask['objects_t']):
        print('No hands or objects detected')
        return None, (None, None)

    nd = time.time()
    print(f"Segment Time: {nd-st:.2f} sec")
    st = nd

    hand_mask, obj_mask, cb_mask = seg_mask['hands'], seg_mask['objects'], seg_mask['cb']
    hand_depth, obj_depth = depth.copy(), depth.copy()
    hand_depth[hand_mask == 0] = 0
    obj_depth[obj_mask == 0] = 0
    
    ##### ///
    ##### For env based depth filter
    # homask = np.logical_or(hand_mask, obj_mask, cb_mask).astype(np.uint8)
    # homask[homask > 0] = 255
    # # filter
    # kernel = np.ones((3,3), np.uint8)
    # homask = cv2.dilate(homask, kernel, iterations=1)
    # depth_env = depth[homask == 0]
    # depth_env[depth_env == 0] = 1000
    # # set depth
    # min_env_depth = min(np.min(depth_env), 0.7)
    # hand_depth[hand_depth > min_env_depth * 1.01] = 0
    # obj_depth[obj_depth > min_env_depth * 1.01] = 0
    #####
    ##### ///
    
    hand_pcd = Depth.sample_pointcloud_open3d(hand_depth)
    obj_pcd = Depth.sample_pointcloud_open3d(obj_depth)

    nd = time.time()
    print(f'Filter Sample Time: {nd-st:.2f} sec')
    st = nd

    return seg_mask, (hand_pcd, obj_pcd)

## visualize result
def visualize(
    frame: np.ndarray,
    depth: np.ndarray,
    seg_mask: Dict,
    hand_pcd: np.ndarray,
    obj_pcd: np.ndarray,
) -> np.ndarray:
    st = time.time()
    # visualize segment
    seg_view = HOSegment.visualize(frame, seg_mask).copy()

    depth_map_gray = Depth.gray(depth, bgr=True)

    h = seg_view.shape[1]
    if h < 480:
        seg_view = cv2.resize(seg_view, dsize=None, fx=480/h, fy=480/h)
        depth_map_gray = cv2.resize(depth_map_gray,dsize=None,fx=480/h,fy=480/h)

    pcd_view = Depth.visualize_pointcloud(
        size=(seg_view.shape[1], seg_view.shape[0]),
        pcds=[np.asarray(hand_pcd.points),np.asarray(obj_pcd.points)],
        rgbs=[(244,194,155),(83,109,254)],
        s=0.5
    )
    
    depth_view = cv2.resize(depth_map_gray,dsize=None,fx=0.3,fy=0.3)
    seg_view[0:depth_view.shape[0], 0:depth_view.shape[1]] = depth_view

    view = cv2.hconcat([seg_view, pcd_view])

    nd = time.time()
    print(f"Visualize Time: {nd-st:.2f} sec")

    return view


def main(args):
    if args.image:
        rgb_list = read_images(args.image)
    if args.video:
        rgb_list = sample_rgb(args.video)
    if args.depth:
        depth_list = read_images(args.depth, depth=True)
    else:
        depth_list = estimate_depth(rgb_list)
    output = Path(args.output)
    # return

    rgb_dir = output / 'rgb'
    depth_dir = output / 'depth'
    handmask_dir = output / 'handmask'
    objmask_dir = output / 'objmask'
    handpcd_dir = output / 'handpcd'
    objpcd_dir = output / 'objpcd'
    view_dir = output / 'view'

    for p in [rgb_dir, depth_dir, handmask_dir, objmask_dir, handpcd_dir, objpcd_dir, view_dir]:
        os.makedirs(p, exist_ok=True)
    
    load_segment_model()
    
    out_vcap = None

    for i in tqdm(range(min(len(rgb_list), len(depth_list)))):
        frame, depth_raw = rgb_list[i], depth_list[i] 

        h, w = frame.shape[:2]

        if h < 480:
            frame = cv2.resize(frame, dsize=None, fx=480/h, fy=480/h, interpolation=cv2.INTER_CUBIC)
            depth_raw = cv2.resize(depth_raw, dsize=None, fx=480/h, fy=480/h, interpolation=cv2.INTER_CUBIC)

        depth_sc = depth_raw.copy().astype(np.float32) * args.depth_scale

        seg_mask, (hand_pcd, object_pcd) = predict(frame, depth_sc)
        if seg_mask is None:
            i = i - 1
            continue

        view = None
        if args.view:
            view = visualize(frame, depth_sc, seg_mask, hand_pcd, object_pcd)
            if args.imshow:
                cv2.imshow('result', view)
                cv2.waitKey(0)

        hm, om = seg_mask['hands'], seg_mask['objects']
        hm[hm > 0] = 255
        om[om > 0] = 255

        if args.out_video:
            if out_vcap is None:
                out_vcap = cv2.VideoWriter(
                    str(view_dir / 'result.mp4'),
                    cv2.VideoWriter_fourcc(*'mp4v'),
                    30,
                    (view.shape[1], view.shape[0])
                )
            out_vcap.write(view)

        cv2.imwrite(str(rgb_dir / f'{i:05d}.png'), frame)
        # save raw 16bit
        cv2.imwrite(str(depth_dir / f'{i:05d}.png'), depth_raw)
        cv2.imwrite(str(handmask_dir / f'{i:05d}.png'), hm)
        cv2.imwrite(str(objmask_dir / f'{i:05d}.png'), om)
        if args.view:
            cv2.imwrite(str(view_dir / f'{i:05d}.png'), view)
        open3d.io.write_point_cloud(str(handpcd_dir / f'{i:05d}.ply'), hand_pcd)
        open3d.io.write_point_cloud(str(objpcd_dir / f'{i:05d}.ply'), object_pcd)

    if out_vcap is not None:
        out_vcap.release()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Predict depth from a video file")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--image", type=str, help="Path to input image files")
    group.add_argument("--video", type=str, help="Path to input video file")
    # parser.add_argument("--save", type=str, default='result')
    parser.add_argument("--depth-scale", type=float, default=1, help="Depth scale factor")
    parser.add_argument("--depth", type=str, help="Path to depth map")
    parser.add_argument("--output", type=str, default='./result')
    parser.add_argument("--view", action='store_true', default=False, help="Save visualization")
    parser.add_argument("--imshow", action='store_true', default=False, help="Show visualization")
    parser.add_argument("--out-video", action='store_true', default=False, help="Save visualization")

    args = parser.parse_args()

    if args.image and args.video:
        raise ValueError("Please specify either --image or --video")

    main(args=args)


