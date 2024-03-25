# Standard Library
from transformers import pipeline
from pathlib import Path
from typing import Any, Optional, Union, List, Tuple
import time

# Third Party
import torch
import cv2
import numpy as np
from PIL import Image
import open3d
import matplotlib.pyplot as plt

class Depth:
    estimate_absolute:bool = False

    def __init__(self, absolute: False) -> None:
        self.estimate_absolute = absolute

        if absolute:
            repo = "isl-org/ZoeDepth"
            # Zoe_NK
            model_zoe_nk = torch.hub.load(repo, "ZoeD_NK", pretrained=True)
            DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
            self.zoe_ = model_zoe_nk.to(DEVICE)

        else:
            self.depth_pipe = pipeline(task="depth-estimation", model="LiheYoung/depth-anything-small-hf")

    @torch.no_grad()
    def infer(
        self,
        image: Union[np.ndarray, str, Path] 
    ) -> np.ndarray:
        if isinstance(image, (str, Path)):
            image = cv2.imread(str(image))
        
        if image is None:
            return None

        if self.estimate_absolute:
            depth = self.zoe_.infer_pil(
                Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            )
        else:    
            depth = self.depth_pipe(
                Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            )["depth"]

        depth = np.array(depth)
        # uint8 to float32
        depth = depth.astype(np.float32)

        if not self.estimate_absolute:
            depth = depth * 2 / 255.0

        return depth

    @staticmethod
    def gray(
        depth: np.ndarray,
        bgr: Optional[bool] = False
    ) -> np.ndarray:
        min_val = np.min(depth)
        max_val = np.max(depth)
        depth = (((depth - min_val) / (max_val - min_val)) * 255).astype(np.uint8)

        if bgr: depth = cv2.cvtColor(depth, cv2.COLOR_GRAY2BGR)
        
        return depth

    @staticmethod
    def predict_intrinsic(
        width: int,
        height: int
    ) -> np.ndarray:
        b = np.max([height, width]) * 1.2
        K = np.array([
            [b, 0, width//2],
            [0, b, height//2],
            [0, 0, 1]
        ], dtype=np.float32)
        return K
    
    @staticmethod
    def sample_pointcloud_open3d(
        depth: np.ndarray,
        K: Optional[Tuple[float,float,int,int]] = None, #(fx,fy,cx,cy)
        trunc: Optional[float] = 1000.0,
        neighbors: Optional[int] = 20,
        std_ratio: Optional[float] = 2.0,
    ) -> open3d.geometry.PointCloud:
        h, w = depth.shape
        if K is None:
            K = (max(w, h) * 1.2, max(w, h), w//2, h//2)
        K = open3d.camera.PinholeCameraIntrinsic(
            width=w, height=h,
            fx = K[0], fy = K[1], cx = K[2], cy = K[3]
        )
        depth = open3d.geometry.Image(depth)
        pcd = open3d.geometry.PointCloud.create_from_depth_image(
            depth, K,
            depth_scale=1.0, depth_trunc=trunc,    
        )
        cl, ind = pcd.remove_statistical_outlier(nb_neighbors=neighbors, std_ratio=std_ratio)
        filtered_pcd = pcd.select_by_index(ind)
        return filtered_pcd

    # @staticmethod
    # def sample_pointclouds_open3d(
    #     depth: np.ndarray,
    #     masks: Union[np.ndarray, List[np.ndarray]] = None,
    #     K: Optional[Tuple[float,float,int,int]] = None, #(fx,fy,cx,cy)
    #     neighbors: Optional[int] = 20,
    #     std_ratio: Optional[float] = 2.0,
    # ) -> Tuple[List[open3d.geometry.PointCloud]]:
    #     if not isinstance(masks, list):
    #         masks = [masks]
    #     h, w = depth.shape
    #     if K is None:
    #         K = (max(w, h) * 1.2, max(w, h), w//2, h//2)
    #     K = open3d.camera.PinholeCameraIntrinsic(
    #         width=w, height=h,
    #         fx = K[0], fy = K[1], cx = K[2], cy = K[3]
    #     )
    #     result = []
    #     for mask in masks:
    #         # select from mask
    #         depth_masked = depth.copy()
    #         depth_masked[mask == 0] = 0

    #         depth_masked = open3d.geometry.Image(depth_masked)
    #         pcd = open3d.geometry.PointCloud.create_from_depth_image(
    #             depth_masked, K,
    #             depth_scale=1.0, depth_trunc=100.0,    
    #         )
    #         cl, ind = pcd.remove_statistical_outlier(nb_neighbors=neighbors, std_ratio=std_ratio)
    #         filtered_pcd = pcd.select_by_index(ind)
    #         result.append(filtered_pcd)

    #     return result

    @staticmethod
    def sample(
        depth: np.ndarray,
        masks: Union[np.ndarray, List[np.ndarray]] = None,
        K: Optional[np.ndarray] = None
    ) -> List[np.ndarray]:
        if not isinstance(masks, list):
            masks = [masks]
        
        h, w = depth.shape

        if K is None:
            K = Depth.predict_intrinsic(w, h)
        
        result = []

        for mask in masks:
            if mask is None:
                mask = np.ones_like(depth)
            mask = mask.astype(np.uint8)
            pcd = Depth.sample_pointcloud(depth, mask, K)
            result.append(pcd)
        
        return result

    @staticmethod
    def filter(
        pcds: List[np.ndarray],
        neighbors: Optional[float] = 20,
        std_ratio: Optional[float] = 2.0
    ) -> List[np.ndarray]:
        result = []
        for points in pcds:
            if len(points) == 0:
                result.append([])
                continue
            pcd = open3d.geometry.PointCloud()
            pcd.points = open3d.utility.Vector3dVector(points)
            cl, ind = pcd.remove_statistical_outlier(nb_neighbors=neighbors, std_ratio=std_ratio)
            inliers = pcd.select_by_index(ind)
            # to numpy
            result.append(np.asarray(inliers.points))
        return result


    @staticmethod
    def sample_pointcloud(
        depth: np.ndarray,
        mask: np.ndarray,
        K : np.ndarray
    ) -> np.ndarray:
        rows, cols = depth.shape
        points = []
        for v in range(rows):
            for u in range(cols):
                d = depth[v, u]
                if d > 0 and mask[v, u] > 0:
                    x = (u - K[0, 2]) * d / K[0, 0]
                    y = (v - K[1, 2]) * d / K[1, 1]
                    z = d
                    points.append([x, y, z])
        points = np.array(points)
        return points
    
    @staticmethod
    def downsample_pointcloud(
        pcd: np.ndarray,
        limit: int
    ) -> np.ndarray:
        if limit < 0: limit = (len(pcd) // (limit * -1))

        # random
        values = np.arange(len(pcd))
        # print(len(pcd))
        idx = np.random.choice(values, limit, replace=False)

        return pcd[idx]


    # @staticmethod
    # def visualize_poincloud(
    #     image: np.ndarray,
    #     pcd: np.ndarray,
    #     K: Optional[np.ndarray] = None,
    #     bgr: Optional[Tuple[int, int, int]] = (0, 255, 0), # bgr
    #     radius: Optional[int] = 0
    # ) -> np.ndarray:
    #     if K is None:
    #         K = Depth.predict_intrinsic(image.shape[1], image.shape[0])

    #     image = image.copy()

    #     for i in range(len(pcd)):
    #         point = pcd[i, :]
    #         point = np.dot(K, point)
    #         z = point[2]
    #         point = point[:2] / z
    #         point = tuple(map(int, point))

    #         # rgb = np.array(color[::-1]) * (z - min_depth) / (d_range)

    #         cv2.circle(image, point, radius, bgr, -1)
    #     return image

    @staticmethod
    def visualize_pointcloud(
        size: Tuple[int, int], # width, height
        pcds: List[np.ndarray],
        rgbs: List[np.ndarray],
        K: Optional[np.ndarray] = None,
        s: Optional[float] = 1,
        out: Optional[str] = 'numpy'
    ) -> Union[np.ndarray, plt.Figure]:
        if K is None:
            K = Depth.predict_intrinsic(size[0], size[1])
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        for i in range(len(pcds)):
            p, c = pcds[i], np.array(rgbs[i]) / 255
            if isinstance(p, open3d.geometry.PointCloud):
                p = np.asarray(p.points)
            if len(p) == 0:
                continue
            ax.scatter(p[:, 0], p[:, 2] , -1 * p[:, 1], s=s, c=[c], marker='o')
        
        ax.view_init(azim=-80, elev=15)
        ax.set_xlabel('X')
        ax.set_ylabel('Z')
        ax.set_zlabel('Y')

        ax.set_xlim(-0.2, 0.2)
        ax.set_ylim(0.3, 0.7)
        ax.set_zlim(-0.3, 0.1)
        # ax.set_aspect('equal')

        # drawing
        if out == 'numpy':
            fig.canvas.draw()
            data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            image = cv2.cvtColor(data, cv2.COLOR_RGB2BGR)

            plt.close(fig)

            # fit to size
            ratio = size[1] / image.shape[0]
            image = cv2.resize(image, dsize=None, fx=ratio, fy=ratio)

            return image
        else:
            return (fig, ax)
