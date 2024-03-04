# Standard Library
from pathlib import Path
from typing import Any, Optional, Union, List, Tuple
import time

# Third Party
import torch
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

class Depth:

    def __init__(self) -> None:
        repo = "isl-org/ZoeDepth"
        # Zoe_N
        model_zoe_nk = torch.hub.load(repo, "ZoeD_NK", pretrained=True)

        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

        self.zoe_ = model_zoe_nk.to(DEVICE)

    @torch.no_grad()
    def infer(
        self,
        image: Union[np.ndarray, str, Path] 
    ) -> np.ndarray:
        if isinstance(image, (str, Path)):
            image = cv2.imread(str(image))
        
        if image is None:
            return None
        
        depth = self.zoe_.infer_pil(
            Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        )

        return depth

    @staticmethod
    def gray(
        depth: np.ndarray,
        bgr: Optional[bool] = False
    ) -> np.ndarray:
        min_val = np.min(depth)
        max_val = np.max(depth)
        depth = ((1 - (depth - min_val) / (max_val - min_val)) * 255).astype(np.uint8)

        if bgr: depth = cv2.cvtColor(depth, cv2.COLOR_GRAY2BGR)
        
        return depth

    @staticmethod
    def predict_intrinsic(
        width: int,
        height: int
    ) -> np.ndarray:
        b = np.max([height, width])
        K = np.array([
            [b, 0, width//2],
            [0, b, height//2],
            [0, 0, 1]
        ], dtype=np.float32)
        return K

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
        s: Optional[float] = 1
    ) -> np.ndarray:
        if K is None:
            K = Depth.predict_intrinsic(size[0], size[1])
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        for i in range(len(pcds)):
            p, c = pcds[i], np.array(rgbs[i]) / 255
            if len(p) == 0:
                continue

            p[:, 1] *= -1
            p[:, 2] *= -1
            ax.scatter(p[:, 2], p[:, 0], p[:, 1], s=s, c=[c], marker='o')
        
        ax.set_xlabel('X')
        ax.set_ylabel('Z')
        ax.set_zlabel('Y')

        # drawing
        fig.canvas.draw()
        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        image = cv2.cvtColor(data, cv2.COLOR_RGB2BGR)

        plt.close(fig)

        # fit to size
        ratio = size[1] / image.shape[0]
        image = cv2.resize(image, dsize=None, fx=ratio, fy=ratio)

        return image
