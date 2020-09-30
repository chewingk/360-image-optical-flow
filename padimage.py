import cv2
import numpy as np
import sys


def padimage(img_in):
    # img_in = cv2.imread(image)
    width = img_in.shape[1]
    pad_width = int(width / 4.0)

    left = img_in[:, 0:pad_width]
    right = img_in[:, width-pad_width:width]

    img_out = np.concatenate([right, img_in, left], axis=1)

    return img_out
