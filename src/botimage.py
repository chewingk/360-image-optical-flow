import cv2
import numpy as np
from numpy import pi
from gettriangleimage import getImageTriangle, getImageBary
import flowio


def botImageFlow(img1, img2, h, w, bary):
    botLine = pi / 2 + np.arctan(0.5)
    # ALL FLIP
    vertexp = list()
    botFlowSet = list()
    # How much speed difference
    # and how much performance difference
    # disflow = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST)
    # disflow = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)
    disflow = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)

    if bary:
        # 67SP
        vertexp.append(np.array([[0.2*pi, botLine],
                                 [0.6*pi, botLine],
                                 [0.4*pi, pi]]))
        # 78SP
        vertexp.append(np.array([[0.6*pi, botLine],
                                 [pi, botLine],
                                 [0.8*pi, pi]]))
        # 89SP
        vertexp.append(np.array([[pi, botLine],
                                 [1.4*pi, botLine],
                                 [1.2*pi, pi]]))
        # 910SP
        vertexp.append(np.array([[1.4*pi, botLine],
                                 [1.8*pi, botLine],
                                 [1.6*pi, pi]]))
        # 106SP
        vertexp.append(np.array([[1.8*pi, botLine],
                                 [2.2*pi, botLine],
                                 [2*pi, pi]]))
    else:
        # 76SP
        vertexp.append(np.array([[0.6*pi, botLine],
                                 [0.2*pi, botLine],
                                 [0.4*pi, pi],
                                 [0.8*pi, pi],
                                 [0, pi]]))

        # 87SP
        vertexp.append(np.array([[pi, botLine],
                                 [0.6*pi, botLine],
                                 [0.8*pi, pi],
                                 [1.2*pi, pi],
                                 [0.4*pi, pi]]))

        # 98SP
        vertexp.append(np.array([[1.4*pi, botLine],
                                 [pi, botLine],
                                 [1.2*pi, pi],
                                 [1.6*pi, pi],
                                 [0.8*pi, pi]]))

        # 109SP
        vertexp.append(np.array([[1.8*pi, botLine],
                                 [1.4*pi, botLine],
                                 [1.6*pi, pi],
                                 [2*pi, pi],
                                 [1.2*pi, pi]]))

        # 610SP
        vertexp.append(np.array([[2.2*pi, botLine],
                                 [1.8*pi, botLine],
                                 [2*pi, pi],
                                 [2.4*pi, pi],
                                 [1.6*pi, pi]]))

    for idx, vertri in enumerate(vertexp):
        if bary:
            out1 = getImageBary(vertri, img1, flip=True)
            out2 = getImageBary(vertri, img2, flip=True)
        else:
            out1 = getImageTriangle(vertri, h, w, img1, flip=True)
            out2 = getImageTriangle(vertri, h, w, img2, flip=True)
        out1uint8 = np.uint8(out1)
        out2uint8 = np.uint8(out2)
        # out1gray = cv2.cvtColor(out1uint8, cv2.COLOR_RGB2GRAY)
        # out2gray = cv2.cvtColor(out2uint8, cv2.COLOR_RGB2GRAY)
        # Calculate flow here
        # curFlow = disflow.calc(out1gray, out2gray, None)
        curFlow = disflow.calc(out1uint8, out2uint8, None)
        botFlowSet.append(curFlow)

        # cv2.imwrite(f'testout/bot{idx}inbary.jpg', out1)
        # cv2.imwrite(f'testout/bot{idx}outbary.jpg', out2)

    return botFlowSet



