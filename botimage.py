import cv2
import numpy as np
from gettriangleimage import getImageTriangle, getImageBary
import flowio


def botImageFlow(img1, img2, h, w, bary):
    # ALL FLIP
    vertexp = list()
    botFlowSet = list()
    # How much speed difference
    # and how much performance difference
    disflow = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST)
    # disflow = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)
    # disflow = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)

    if bary:
        # 67SP
        vertexp.append(np.array([[0.2*np.pi, np.pi/1.5],
                                 [0.6*np.pi, np.pi/1.5],
                                 [0.4*np.pi, np.pi]]))
        # 78SP
        vertexp.append(np.array([[0.6*np.pi, np.pi/1.5],
                                 [np.pi, np.pi/1.5],
                                 [0.8*np.pi, np.pi]]))
        # 89SP
        vertexp.append(np.array([[np.pi, np.pi/1.5],
                                 [1.4*np.pi, np.pi/1.5],
                                 [1.2*np.pi, np.pi]]))
        # 910SP
        vertexp.append(np.array([[1.4*np.pi, np.pi/1.5],
                                 [1.8*np.pi, np.pi/1.5],
                                 [1.6*np.pi, np.pi]]))
        # 106SP
        vertexp.append(np.array([[1.8*np.pi, np.pi/1.5],
                                 [2.2*np.pi, np.pi/1.5],
                                 [2*np.pi, np.pi]]))
    else:
        # 76SP
        vertexp.append(np.array([[0.6*np.pi, np.pi/1.5],
                                 [0.2*np.pi, np.pi/1.5],
                                 [0.4*np.pi, np.pi],
                                 [0.8*np.pi, np.pi],
                                 [0, np.pi]]))

        # 87SP
        vertexp.append(np.array([[np.pi, np.pi/1.5],
                                 [0.6*np.pi, np.pi/1.5],
                                 [0.8*np.pi, np.pi],
                                 [1.2*np.pi, np.pi],
                                 [0.4*np.pi, np.pi]]))

        # 98SP
        vertexp.append(np.array([[1.4*np.pi, np.pi/1.5],
                                 [np.pi, np.pi/1.5],
                                 [1.2*np.pi, np.pi],
                                 [1.6*np.pi, np.pi],
                                 [0.8*np.pi, np.pi]]))

        # 109SP
        vertexp.append(np.array([[1.8*np.pi, np.pi/1.5],
                                 [1.4*np.pi, np.pi/1.5],
                                 [1.6*np.pi, np.pi],
                                 [2*np.pi, np.pi],
                                 [1.2*np.pi, np.pi]]))

        # 610SP
        vertexp.append(np.array([[2.2*np.pi, np.pi/1.5],
                                 [1.8*np.pi, np.pi/1.5],
                                 [2*np.pi, np.pi],
                                 [2.4*np.pi, np.pi],
                                 [1.6*np.pi, np.pi]]))

    for idx, vertri in enumerate(vertexp):
        if bary:
            out1 = getImageBary(vertri, img1, flip=True)
            out2 = getImageBary(vertri, img2, flip=True)
        else:
            out1 = getImageTriangle(vertri, h, w, img1, flip=True)
            out2 = getImageTriangle(vertri, h, w, img2, flip=True)
        out1uint8 = np.uint8(out1)
        out2uint8 = np.uint8(out2)
        out1gray = cv2.cvtColor(out1uint8, cv2.COLOR_RGB2GRAY)
        out2gray = cv2.cvtColor(out2uint8, cv2.COLOR_RGB2GRAY)
        # Calculate flow here
        curFlow = disflow.calc(out1gray, out2gray, None)
        botFlowSet.append(curFlow)

        # cv2.imwrite(f'testout/bot{idx}inbary.jpg', out1)
        # cv2.imwrite(f'testout/bot{idx}outbary.jpg', out2)

    return botFlowSet



