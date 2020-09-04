import cv2
import numpy as np
from gettriangleimage import getImageTriangle, getImageBary
import flowio
import flow_vis as fv


def topImageFlow(img1, img2, h, w, bary):
    vertexp = list()
    topFlowSet = list()
    # How much speed difference
    # and how much performance difference
    disflow = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST)
    # disflow = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)
    # disflow = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    if bary:
        # 12NP
        vertexp.append(np.array([[0.2*np.pi, 0],
                                 [0, np.pi/3],
                                 [0.4*np.pi, np.pi/3]]))
        # 23NP
        vertexp.append(np.array([[0.6*np.pi, 0],
                                 [0.4*np.pi, np.pi/3],
                                 [0.8*np.pi, np.pi/3]]))
        # 34NP
        vertexp.append(np.array([[np.pi, 0],
                                 [0.8*np.pi, np.pi/3],
                                 [1.2*np.pi, np.pi/3]]))
        # 45NP
        vertexp.append(np.array([[1.4*np.pi, 0],
                                 [1.2*np.pi, np.pi/3],
                                 [1.6*np.pi, np.pi/3]]))
        # 51NP
        vertexp.append(np.array([[1.8*np.pi, 0],
                                 [1.6*np.pi, np.pi/3],
                                 [2*np.pi, np.pi/3]]))
    else:
        # 12NP
        vertexp.append(np.array([[0, np.pi/3],
                                 [0.4*np.pi, np.pi/3],
                                 [0.2*np.pi, 0],
                                 [-0.2*np.pi, 0],
                                 [0.6*np.pi, 0]]))
        # 23NP
        vertexp.append(np.array([[0.4*np.pi, np.pi/3],
                                 [0.8*np.pi, np.pi/3],
                                 [0.6*np.pi, 0],
                                 [0.2*np.pi, 0],
                                 [np.pi, 0]]))
        # 34NP
        vertexp.append(np.array([[0.8*np.pi, np.pi/3],
                                 [1.2*np.pi, np.pi/3],
                                 [np.pi, 0],
                                 [0.6*np.pi, 0],
                                 [1.4*np.pi, 0]]))
        # 45NP
        vertexp.append(np.array([[1.2*np.pi, np.pi/3],
                                 [1.6*np.pi, np.pi/3],
                                 [1.4*np.pi, 0],
                                 [np.pi, 0],
                                 [1.8*np.pi, 0]]))
        # 51NP
        vertexp.append(np.array([[1.6*np.pi, np.pi/3],
                                 [2*np.pi, np.pi/3],
                                 [1.8*np.pi, 0],
                                 [1.4*np.pi, 0],
                                 [2.2*np.pi, 0]]))

    for idx, vertri in enumerate(vertexp):
        if bary:
            out1 = getImageBary(vertri, img1)
            out2 = getImageBary(vertri, img2)
        else:
            out1 = getImageTriangle(vertri, h, w, img1)
            out2 = getImageTriangle(vertri, h, w, img2)
        out1uint8 = np.uint8(out1)
        out2uint8 = np.uint8(out2)
        out1gray = cv2.cvtColor(out1uint8, cv2.COLOR_RGB2GRAY)
        out2gray = cv2.cvtColor(out2uint8, cv2.COLOR_RGB2GRAY)
        # Calculate flow here
        curFlow = disflow.calc(out1gray, out2gray, None)
        topFlowSet.append(curFlow)
        # mmm = fv.flow_to_color(curFlow)
        # cv2.imwrite(f"testout/top{idx}flow.jpg")

        # cv2.imwrite(f'testout/top{idx}in.jpg', out1)
        # cv2.imwrite(f'testout/top{idx}out.jpg', out2)

    return topFlowSet
