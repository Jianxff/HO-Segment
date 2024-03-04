# Standard Library
from pathlib import Path
import os
from typing import Optional, Union, Any

# Third Party
import torch
import cv2
import numpy as np
import argparse

# MM Segmentation
import mmcv
from mmseg.apis import inference_segmentor, init_segmentor, InferencePipe

class Pipeline:
    def __init__(
        self,
        work_dirs: Union[str, Path],
        device: Optional[str] = 'cuda:0'
    ) -> None:
        self.load(
            work_dirs=work_dirs,
            device=device
        )
        

    def load(
        self,
        work_dirs: Union[str, Path],
        device: Optional[str] = 'cuda:0'
    ) -> Any:
        work_dirs = Path(work_dirs)

        # build the model from a config file and a checkpoint file
        self.pipe_hands = InferencePipe(
            model = init_segmentor(
                config=str(work_dirs / 'seg_twohands_ccda' / 'seg_twohands_ccda.py'),
                checkpoint=str(work_dirs / 'seg_twohands_ccda' / 'best_mIoU_iter_56000.pth'),
                device=device
            )
        )

        self.pipe_cb = InferencePipe(
            model = init_segmentor(
                config=str(work_dirs / 'twohands_to_cb_ccda' / 'twohands_to_cb_ccda.py'),
                checkpoint=str(work_dirs / 'twohands_to_cb_ccda' / 'best_mIoU_iter_76000.pth'),
                device=device
            )
        )

        self.pipe_obj = InferencePipe(
            model = init_segmentor(
                config=str(work_dirs / 'twohands_cb_to_obj1_ccda' / 'twohands_cb_to_obj1_ccda.py'),
                checkpoint=str(work_dirs / 'twohands_cb_to_obj1_ccda' / 'best_mIoU_iter_34000.pth'),
                device=device
            )
        )

        return self

    @torch.no_grad()
    def infer(
        self,
        image: np.ndarray,
        hands_only: Optional[bool] = False
    ) -> dict[str, np.ndarray]:
        # inference
        seg_result_hands = self.pipe_hands(image)

        if hands_only:
            return {
                'hands': seg_result_hands,
                'objects': np.zeros_like(seg_result_hands)
            }

        seg_result_cb = self.pipe_cb(image, seg_result_hands)
        seg_result_obj = self.pipe_obj(image, seg_result_hands, seg_result_cb)

        return {
            'hands': seg_result_hands, 
            'objects': seg_result_obj
        }

    def __call__(
        self,
        image: Union[np.ndarray, str, Path]
    ) -> dict[str, np.ndarray]:
        if isinstance(image, (str, Path)):
            image = cv2.imread(str(image))

        return self.infer(image)

    @staticmethod
    def visualize(
        image: np.ndarray,
        result: dict[str, np.ndarray],
        save: Optional[Union[str, Path]] = None
    ) -> np.ndarray:
        seg_hands = result['hands']
        seg_objs = result['objects']

        # visualize
        seg_result = seg_hands.copy()
        seg_result[seg_objs == 1] = 3
        seg_result[seg_objs == 2] = 4
        seg_result[seg_objs == 3] = 5

        # colorize
        alpha = 0.4
        seg_color = np.zeros((image.shape))
        seg_color[seg_result == 0] = (0,    0,   0)     # background
        seg_color[seg_result == 1] = (255,  0,   0)     # left_hand
        seg_color[seg_result == 2] = (0,    0,   255)   # right_hand
        seg_color[seg_result == 3] = (255,  0,   255)   # left_object1
        seg_color[seg_result == 4] = (0,    255, 255)   # right_object1
        seg_color[seg_result == 5] = (0,    255, 0)     # two_object1
        vis = image * (1 - alpha) + seg_color * alpha
        vis = vis.astype(np.uint8)
        
        # save
        if save:
            cv2.imwrite(str(save), vis)
        
        return vis