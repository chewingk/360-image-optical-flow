import numpy as np
import cv2

from visfunction import bilinear_interpolation


def endPointError(truthFlow, optFlow):
    sumEPE = 0
    height = truthFlow.shape[0]
    width = truthFlow.shape[1]
    for row in range(height):
        for col in range(width):
            tf = truthFlow[row][col]
            of = optFlow[row][col]
            epe = np.linalg.norm(tf - of)
            # epe = np.sqrt((tf[0] - of[0])**2 + (tf[1] - of[1])**2)
            sumEPE = sumEPE + epe
    return sumEPE / (width * height)


def angularError(truthFlow, optFlow):
    sumAE = 0
    height = truthFlow.shape[0]
    width = truthFlow.shape[1]
    for row in range(height):
        for col in range(width):
            # ae = abs(truthFlow[row][col][1] - optFlow[row][col][1])
            tf = truthFlow[row][col]
            of = optFlow[row][col]
            np.append(tf, 0)
            np.append(of, 0)
            ae = np.degrees(angle_between(tf, of))
            sumAE = sumAE + ae
    return sumAE / (width * height)


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
