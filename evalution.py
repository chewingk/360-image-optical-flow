import numpy as np
import cv2

from visfunction import bilinear_interpolation


def endPointError(truthFlow, optFlow):
    diffFlow = truthFlow - optFlow
    diffNorm = np.linalg.norm(diffFlow, axis=2)
    return np.average(diffNorm)


def angularError(truthFlow, optFlow):
    dotPro = np.sum(truthFlow*optFlow, axis=2)
    ae = dotPro / (np.linalg.norm(truthFlow, axis=2) * np.linalg.norm(optFlow, axis=2))
    return np.rad2deg(np.nanmean(np.arccos(ae)))


def frameInterpolation(optFlow, originalImage):
    rowSize, colSize, rgbSize = originalImage.shape
    # rowSize, colSize = originalImage.shape
    interpolatedImage = np.zeros([rowSize, colSize, rgbSize])
    # interpolatedImage = np.zeros([rowSize, colSize])
    for row in range(rowSize):
        for col in range(colSize):
            # delta = pol2cart(0.5*optFlow[row][col][0], optFlow[row][col][1])
            targetRow = row + 0.5 * optFlow[row][col][1]
            targetCol = col + 0.5 * optFlow[row][col][0]
            if (0 <= targetCol < colSize-1) and (0 <= targetRow < rowSize-1):
                bVals = bilinear_interpolation(targetRow, targetCol)
                for rgb in range(rgbSize):
                    interpolatedImage[row][col][rgb] = originalImage[int(bVals[0])][int(bVals[2])][rgb] * bVals[4] + \
                                                       originalImage[int(bVals[0])][int(bVals[3])][rgb] * bVals[5] + \
                                                       originalImage[int(bVals[1])][int(bVals[2])][rgb] * bVals[6] + \
                                                       originalImage[int(bVals[1])][int(bVals[3])][rgb] * bVals[7]
                # interpolatedImage[row][col] = originalImage[int(bVals[0])][int(bVals[2])] * bVals[4] + \
                #                               originalImage[int(bVals[0])][int(bVals[3])] * bVals[5] + \
                #                               originalImage[int(bVals[1])][int(bVals[2])] * bVals[6] + \
                #                               originalImage[int(bVals[1])][int(bVals[3])] * bVals[7]
    return interpolatedImage


def frameInterpolationError(interpolatedImage, groundTruthImage):
    diff = np.subtract(groundTruthImage, interpolatedImage) / 255.0
    interpolatedError = np.linalg.norm(diff)
    return interpolatedError


def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return np.array([rho, phi])


def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return np.array([x, y])


def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def flow_correction(flow):
    # flow is in UV
    # u is hori
    # v is vert
    h, w = flow.shape[:2]
    for row in range(h):
        for col in range(w):
            hori = flow[row][col][0]
            vert = flow[row][col][1]
            # print(hori)
            if abs(hori) > w/2.0:
                # print('hihi')
                flow[row][col][0] = hori - w if hori > 0 else hori + w
            if abs(vert) > h/2.0:
                flow[row][col][1] = vert - h if vert > 0 else vert + h
    return flow
