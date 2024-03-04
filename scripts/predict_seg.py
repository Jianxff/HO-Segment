# Standard Library
from pathlib import Path
import argparse

# Third Party
import cv2

# HO Segment
from modules.pipeline import Pipeline


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--work_dirs", default='./work_dirs', type=str)
    parser.add_argument("--image", type=str)
    parser.add_argument("--save", default='result.png', type=str)

    args = parser.parse_args()

    pipe = Pipeline(work_dirs=args.work_dirs)

    seg = pipe(image=args.image)

    pipe.visualize(
        image=cv2.imread(args.image),
        result=seg,
        save=args.save
    )