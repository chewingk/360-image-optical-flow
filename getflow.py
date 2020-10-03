import flowio
import cv2
import numpy as np
from topimage import topImageFlow
from midimage import midImageFlow
from botimage import botImageFlow


def getFlowSetPreProcessed(img1, img2, triangleFlowSet):
    # triangleFlowSet = np.load('flowSet.npy')
    disflow = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    flowHeight, flowWidth = triangleFlowSet.shape[1:3]
    flowSet = list()
    for flowIdx in range(20):
        triangleFlow = triangleFlowSet[flowIdx]
        curImg1 = np.zeros([flowHeight, flowWidth])
        curImg2 = np.zeros([flowHeight, flowWidth])
        for row in range(flowHeight):
            for col in range(flowWidth):
                curBiVal = triangleFlow[row, col]
                # print(curBiVal)
                curImg1[row, col] = img1[int(curBiVal[0])][int(curBiVal[2])] * curBiVal[4] + \
                    img1[int(curBiVal[0])][int(curBiVal[3])] * curBiVal[5] + \
                    img1[int(curBiVal[1])][int(curBiVal[2])] * curBiVal[6] + \
                    img1[int(curBiVal[1])][int(curBiVal[3])] * curBiVal[7]
                curImg2[row, col] = img2[int(curBiVal[0])][int(curBiVal[2])] * curBiVal[4] + \
                    img2[int(curBiVal[0])][int(curBiVal[3])] * curBiVal[5] + \
                    img2[int(curBiVal[1])][int(curBiVal[2])] * curBiVal[6] + \
                    img2[int(curBiVal[1])][int(curBiVal[3])] * curBiVal[7]
        # print(curImg1.shape)
        curImg1uint8 = np.uint8(curImg1)
        curImg2uint8 = np.uint8(curImg2)
        curFlow = disflow.calc(curImg1uint8, curImg2uint8, None)
        flowSet.append(curFlow)
    return flowSet


def getFlowSet(img1, img2, h, w, bary):
    # h and w are the height and width of flow images
    # not original equi image
    flowSet = list()
    flowSet.append(topImageFlow(img1, img2, h, w, bary))
    flowSet.append(midImageFlow(img1, img2, h, w, bary))
    flowSet.append(botImageFlow(img1, img2, h, w, bary))
    return flowSet


def getFlowFileSet(dirName):
    flowSet = []
    topFlowSet = []
    midFlowSet = []
    botFlowSet = []
    # Get NP flows
    for i in range(5):
        tempFlow = flowio.readFlowFile(f"{dirName}/top{i}.flo")
        topFlowSet.append(tempFlow)
    for i in range(10):
        tempFlow = flowio.readFlowFile(f"{dirName}/mid{i}.flo")
        midFlowSet.append(tempFlow)
    for i in range(5):
        tempFlow = flowio.readFlowFile(f"{dirName}/bot{i}.flo")
        botFlowSet.append(tempFlow)
    flowSet.append(topFlowSet)
    flowSet.append(midFlowSet)
    flowSet.append(botFlowSet)

    return flowSet
