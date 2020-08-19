import cv2
import numpy as np
from gettriangleimage import getImageTriangle
import flowio


def midImageFlow(img1, img2, h, w):
    # I have a equi image
    # h = 200
    # w = 200
    # Define output directory
    # outDir = ""

    # readin1 = cv2.imread('360images/frame145.jpg')
    # readin2 = cv2.imread('360images/frame147.jpg')
    # readin1 = cv2.imread("optical_flow_gt/0001_rgb.jpg")
    # readin2 = cv2.imread("optical_flow_gt/0002_rgb.jpg")

    vertexp = list()
    midFlowSet = list()
    # How much speed difference
    # and how much performance difference
    disflow = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST)
    # disflow = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)
    # disflow = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    # 121067----------------------------------- 0 Flip
    vertexp.append(np.array([[0.4*np.pi, np.pi/3],
                            [0, np.pi/3],
                            [0.2*np.pi, np.pi/1.5],
                            [0.6*np.pi, np.pi/1.5],
                            [-0.2*np.pi, np.pi/1.5]]))
    # 12367---------------------------------- 1
    vertexp.append(np.array([[0.2*np.pi, np.pi/1.5],
                            [0.6*np.pi, np.pi/1.5],
                            [0.4*np.pi, np.pi/3],
                            [0, np.pi/3],
                            [0.8*np.pi, np.pi/3]]))
    # 23678---------------------------------- 2 Flip
    vertexp.append(np.array([[0.8*np.pi, np.pi/3],
                            [0.4*np.pi, np.pi/3],
                            [0.6*np.pi, np.pi/1.5],
                            [np.pi, np.pi/1.5],
                            [0.2*np.pi, np.pi/1.5]]))
    # 78234---------------------------------- 3
    vertexp.append(np.array([[0.6*np.pi, np.pi/1.5],
                            [np.pi, np.pi/1.5],
                            [0.8*np.pi, np.pi/3],
                            [0.4*np.pi, np.pi/3],
                            [1.2*np.pi, np.pi/3]]))
    # 34789---------------------------------- 4 Flip
    vertexp.append(np.array([[1.2*np.pi, np.pi/3],
                            [0.8*np.pi, np.pi/3],
                            [np.pi, np.pi/1.5],
                            [1.4*np.pi, np.pi/1.5],
                            [0.6*np.pi, np.pi/1.5]]))
    # 34589----------------------------------- 5
    vertexp.append(np.array([[np.pi, np.pi/1.5],
                            [1.4*np.pi, np.pi/1.5],
                            [1.2*np.pi, np.pi/3],
                            [0.8*np.pi, np.pi/3],
                            [1.6*np.pi, np.pi/3]]))
    # 458910----------------------------------- 6 Flip
    vertexp.append(np.array([[1.6*np.pi, np.pi/3],
                            [1.2*np.pi, np.pi/3],
                            [1.4*np.pi, np.pi/1.5],
                            [1.8*np.pi, np.pi/1.5],
                            [np.pi, np.pi/1.5]]))
    # 451910----------------------------------- 7
    vertexp.append(np.array([[1.4*np.pi, np.pi/1.5],
                            [1.8*np.pi, np.pi/1.5],
                            [1.6*np.pi, np.pi/3],
                            [1.2*np.pi, np.pi/3],
                            [2*np.pi, np.pi/3]]))
    # 519106----------------------------------- 8 Flip
    vertexp.append(np.array([[2*np.pi, np.pi/3],
                            [1.6*np.pi, np.pi/3],
                            [1.8*np.pi, np.pi/1.5],
                            [2.2*np.pi, np.pi/1.5],
                            [1.4*np.pi, np.pi/1.5]]))
    # 106512----------------------------------- 9
    vertexp.append(np.array([[1.8*np.pi, np.pi/1.5],
                            [2.2*np.pi, np.pi/1.5],
                            [2*np.pi, np.pi/3],
                            [1.6*np.pi, np.pi/3],
                            [2.4*np.pi, np.pi/3]]))

    for idx, vertri in enumerate(vertexp):
        flip = (idx % 2 == 0)
        out1 = getImageTriangle(vertri, h, w, img1, flip=flip)
        out2 = getImageTriangle(vertri, h, w, img2, flip=flip)
        out1uint8 = np.uint8(out1)
        out2uint8 = np.uint8(out2)
        out1gray = cv2.cvtColor(out1uint8, cv2.COLOR_RGB2GRAY)
        out2gray = cv2.cvtColor(out2uint8, cv2.COLOR_RGB2GRAY)
        # Calculate flow here
        curFlow = disflow.calc(out1gray, out2gray, None)
        # Write flow to flo
        # flowio.writeFlowFile(curFlow, f"{outDir}/mid{idx}.flo")
        # cv2.imwrite(f'testout/{idx}in.jpg', out1)
        # cv2.imwrite(f'testout/{idx}out.jpg', out2)
        midFlowSet.append(curFlow)

    return midFlowSet
