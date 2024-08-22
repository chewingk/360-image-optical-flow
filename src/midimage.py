import cv2
import numpy as np
from numpy import pi
from gettriangleimage import getImageTriangle, getImageBary
import flowio


def midImageFlow(img1, img2, h, w, bary):
    # I have a equi image
    # h = 200
    # w = 200
    # Define output directory
    # outDir = ""

    # readin1 = cv2.imread('360images/frame145.jpg')
    # readin2 = cv2.imread('360images/frame147.jpg')
    # readin1 = cv2.imread("optical_flow_gt/0001_rgb.jpg")
    # readin2 = cv2.imread("optical_flow_gt/0002_rgb.jpg")

    topLine = pi / 2 - np.arctan(0.5)
    botLine = pi / 2 + np.arctan(0.5)
    vertexp = list()
    midFlowSet = list()
    # How much speed difference
    # and how much performance difference
    # disflow = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST)
    # disflow = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)
    disflow = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    if bary:
        # 016--------------------------------------0 Flip
        vertexp.append(np.array([[0, topLine],
                                 [0.4*pi, topLine],
                                 [0.2*pi, botLine]]))
        # 167--------------------------------------1
        vertexp.append(np.array([[0.4*pi, topLine],
                                 [0.2*pi, botLine],
                                 [0.6*pi, botLine]]))
        # 127 --------------------------------------2 Flip
        vertexp.append(np.array([[0.4*pi, topLine],
                                 [0.8*pi, topLine],
                                 [0.6*pi, botLine]]))
        # 278-------------------------------------3
        vertexp.append(np.array([[0.8*pi, topLine],
                                 [0.6*pi, botLine],
                                 [pi, botLine]]))
        # 238-------------------------------------4 Flip
        vertexp.append(np.array([[0.8*pi, topLine],
                                 [1.2*pi, topLine],
                                 [pi, botLine]]))
        # 389------------------------------5
        vertexp.append(np.array([[1.2*pi, topLine],
                                 [pi, botLine],
                                 [1.4*pi, botLine]]))
        # 349 ---------------------------------6 Flip
        vertexp.append(np.array([[1.2*pi, topLine],
                                 [1.6*pi, topLine],
                                 [1.4*pi, botLine]]))
        # 4910 -------------------------------- 7
        vertexp.append(np.array([[1.6*pi, topLine],
                                 [1.4*pi, botLine],
                                 [1.8*pi, botLine]]))
        # 4510 -------------------------------------8 Flip
        vertexp.append(np.array([[1.6*pi, topLine],
                                 [2*pi, topLine],
                                 [1.8*pi, botLine]]))
        # 5106 ----------------------------------9
        vertexp.append(np.array([[2*pi, topLine],
                                 [1.8*pi, botLine],
                                 [2.2*pi, botLine]]))
    else:
        # 121067----------------------------------- 0 Flip
        vertexp.append(np.array([[0.4*pi, topLine],
                                [0, topLine],
                                [0.2*pi, botLine],
                                [0.6*pi, botLine],
                                [-0.2*pi, botLine]]))
        # 12367---------------------------------- 1
        vertexp.append(np.array([[0.2*pi, botLine],
                                [0.6*pi, botLine],
                                [0.4*pi, topLine],
                                [0, topLine],
                                [0.8*pi, topLine]]))
        # 23678---------------------------------- 2 Flip
        vertexp.append(np.array([[0.8*pi, topLine],
                                [0.4*pi, topLine],
                                [0.6*pi, botLine],
                                [pi, botLine],
                                [0.2*pi, botLine]]))
        # 78234---------------------------------- 3
        vertexp.append(np.array([[0.6*pi, botLine],
                                [pi, botLine],
                                [0.8*pi, topLine],
                                [0.4*pi, topLine],
                                [1.2*pi, topLine]]))
        # 34789---------------------------------- 4 Flip
        vertexp.append(np.array([[1.2*pi, topLine],
                                [0.8*pi, topLine],
                                [pi, botLine],
                                [1.4*pi, botLine],
                                [0.6*pi, botLine]]))
        # 34589----------------------------------- 5
        vertexp.append(np.array([[pi, botLine],
                                [1.4*pi, botLine],
                                [1.2*pi, topLine],
                                [0.8*pi, topLine],
                                [1.6*pi, topLine]]))
        # 458910----------------------------------- 6 Flip
        vertexp.append(np.array([[1.6*pi, topLine],
                                [1.2*pi, topLine],
                                [1.4*pi, botLine],
                                [1.8*pi, botLine],
                                [pi, botLine]]))
        # 451910----------------------------------- 7
        vertexp.append(np.array([[1.4*pi, botLine],
                                [1.8*pi, botLine],
                                [1.6*pi, topLine],
                                [1.2*pi, topLine],
                                [2*pi, topLine]]))
        # 519106----------------------------------- 8 Flip
        vertexp.append(np.array([[2*pi, topLine],
                                [1.6*pi, topLine],
                                [1.8*pi, botLine],
                                [2.2*pi, botLine],
                                [1.4*pi, botLine]]))
        # 106512----------------------------------- 9
        vertexp.append(np.array([[1.8*pi, botLine],
                                [2.2*pi, botLine],
                                [2*pi, topLine],
                                [1.6*pi, topLine],
                                [2.4*pi, topLine]]))

    for idx, vertri in enumerate(vertexp):
        flip = (idx % 2 == 0)
        if bary:
            out1 = getImageBary(vertri, img1, flip=flip)
            out2 = getImageBary(vertri, img2, flip=flip)
        else:
            out1 = getImageTriangle(vertri, h, w, img1, flip=flip)
            out2 = getImageTriangle(vertri, h, w, img2, flip=flip)
        out1uint8 = np.uint8(out1)
        out2uint8 = np.uint8(out2)
        # out1gray = cv2.cvtColor(out1uint8, cv2.COLOR_RGB2GRAY)
        # out2gray = cv2.cvtColor(out2uint8, cv2.COLOR_RGB2GRAY)
        # Calculate flow here
        # curFlow = disflow.calc(out1gray, out2gray, None)
        curFlow = disflow.calc(out1uint8, out2uint8, None)
        # Write flow to flo
        # flowio.writeFlowFile(curFlow, f"{outDir}/mid{idx}.flo")
        # cv2.imwrite(f'testout/{idx}in.jpg', out1)
        # cv2.imwrite(f'testout/{idx}out.jpg', out2)
        midFlowSet.append(curFlow)

        # cv2.imwrite(f'testout/{idx}inbary.jpg', out1)
        # cv2.imwrite(f'testout/{idx}outbary.jpg', out2)

    return midFlowSet
