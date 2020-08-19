import cv2
import numpy as np


def bilinear_interpolation(r, c, h, w):
    rFloor = np.floor(r)
    rCeil = np.ceil(r)
    cFloor = np.floor(c)
    cCeil = np.ceil(c)
    beta = r - rFloor
    alpha = c - cFloor
    top_left = (1 - alpha) * (1 - beta)
    top_right = alpha * (1 - beta)
    bot_left = (1 - alpha) * beta
    bot_right = alpha * beta

    if rFloor >= h:
        rFloor = rFloor - h
    elif rFloor < 0:
        rFloor = rFloor + h
    if rCeil >= h:
        rCeil = rCeil - h
    elif rCeil < 0:
        rCeil = rCeil + h
    if cFloor >= w:
        cFloor = cFloor - w
    elif cFloor < 0:
        cFloor = cFloor + w
    if cCeil >= w:
        cCeil = cCeil - w
    elif cCeil < 0:
        cCeil = cCeil + w

    return np.array([rFloor, rCeil, cFloor, cCeil, top_left, top_right, bot_left, bot_right])


def backward_mapping(srcImg, dstRowSize, dstColSize, h):
    # Assume it to be colour image
    rowSize, colSize, rgbSize = srcImg.shape
    dstImg = np.empty(dtype=int)
    for r in range(dstRowSize):
        for c in range(dstColSize):
            dstPt = np.array([c, r, 1]).T
            srcPt = np.multiply(h, dstPt)
            srcPtNorm = srcPt / srcPt[2]
            if srcPtNorm[1] >= 0 & srcPtNorm[1] <= rowSize-1 & srcPtNorm[0] >= 0 & srcPtNorm[0] <= colSize-1:
                vals = bilinear_interpolation(srcPtNorm[1], srcPtNorm[0])
                for rgb in range(rgbSize):
                    dstImg[r][c][rgb] = srcImg[vals[0]][vals[2]][rgb] * vals[4] + \
                                        srcImg[vals[0]][vals[3]][rgb] * vals[5] + \
                                        srcImg[vals[1]][vals[2]][rgb] * vals[6] + \
                                        srcImg[vals[1]][vals[3]][rgb] * vals[7]
            else:
                for rgb in range(rgbSize):
                    dstImg[r][c][rgb] = 255
    return dstImg
