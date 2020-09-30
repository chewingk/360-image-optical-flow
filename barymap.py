import cv2
import numpy as np
from scipy.spatial import Delaunay
from equiprojection import sphere2equi, newcart2bary, equiCoor2bary, equibaryPreProcess, bary2equiCoor
from visfunction import bilinear_interpolation
from getflow import getFlowSet


def equiImg2EquiFlowBary(img1, img2):
    h = img1.shape[0]
    w = img2.shape[1]
    topMar = (np.pi/2 - np.arctan(0.5)) / np.pi * h
    botMar = (np.pi/2 + np.arctan(0.5)) / np.pi * h
    finalFlow = np.zeros([h, w, 2])
    equiVertexArray = getMidVertexArray(h, w)
    tri = Delaunay(equiVertexArray)
    triArray = tri.simplices
    # flowHeight = 240
    # flowWidth = 240
    flowHeight = 300
    flowWidth = 300
    flowSet = getFlowSet(img1, img2, flowHeight, flowWidth, True)
    # There are padding in the cropped images
    # triOrientation = np.array([[[19, 119], [219, 19], [219, 219]],
    #                            [[19, 19], [19, 219], [219, 119]]])
    triOrientation = np.array([[[49, 149], [249, 49], [249, 249]],
                               [[49, 49], [49, 249], [249, 149]]])

    for y in range(h):
        for x in range(w):
            # NP
            if y < topMar:
                rectImage = detRect(x, w, True)
                rectVertex = getRectVertex(rectImage, True, h, w)
                curX = x
                curY = y
                p = np.array([curY, curX])
                finalFlow[y][x] = findFlowValue(h, w, p, rectVertex[0], rectVertex[1], rectVertex[2],
                                                triOrientation[0][0], triOrientation[0][1], triOrientation[0][2],
                                                flowSet[0][rectImage])
            # SP
            elif y >= botMar:
                rectImage = detRect(x, w, False)
                curX = x
                curY = y
                # It can be negative (on the left)
                # Should add w to x when doing warping
                # i.e. x = x + w
                if rectImage < 0:
                    rectImage = rectImage + 5
                    curX = curX + w
                rectVertex = getRectVertex(rectImage, False, h, w)
                p = np.array([curY, curX])
                finalFlow[y][x] = findFlowValue(h, w, p, rectVertex[0], rectVertex[1], rectVertex[2],
                                                triOrientation[1][0], triOrientation[1][1], triOrientation[1][2],
                                                flowSet[2][rectImage])
            # Mid
            # The same as non-bary
            elif topMar <= y < botMar:
                p = np.array([y, x])
                index = tri.find_simplex(p)
                # Find which image this pixel belongs to
                triImage = detTri(triArray[index])
                finalFlow[y][x] = getMidFinalFlow(h, w, p, flowSet, triImage, equiVertexArray, triOrientation)

    return finalFlow


# def equiImg2EquiFlow(img1, img2):
#     h = img1.shape[0]
#     w = img1.shape[1]
#     finalFlow = np.zeros((h, w, 2))
#     # For mid triangles
#     # POINTS SHOULD BE CONVERTED FROM SPHERE COORDINATE!!!!!!!!!!!!!!!!!!!
#     # topThi = np.pi / 3
#     # botThi = np.pi / 1.5
#     # intervalTheta = np.pi / 5
#     # equiVertexArray = np.zeros([13, 2])
#     # for vIdx in range(6):
#     #     equiVertexArray[vIdx] = sphere2equi(h, w, np.array([vIdx*2*intervalTheta, topThi]))
#     # for vIdx in range(6, 11):
#     #     equiVertexArray[vIdx] = sphere2equi(h, w, np.array([((vIdx-6)*2+1)*intervalTheta, botThi]))
#     # equiVertexArray[11] = sphere2equi(h, w, np.array([-1*intervalTheta, botThi]))
#     # equiVertexArray[12] = sphere2equi(h, w, np.array([11*intervalTheta, botThi]))
#     equiVertexArray = getMidVertexArray(h, w)
#     flowHeight = 200
#     flowWidth = 200
#     flowSet = getFlowSet(img1, img2, flowHeight, flowWidth, False)
#     # For top and bot
#     homographySet = getHomographySet(h, w, flowHeight, flowWidth)
#     triOrientation = np.array([[[0, 0], [0, flowWidth-1], [flowHeight-1, int(flowWidth/2)]],
#                                [[0, int(flowWidth/2)], [flowHeight-1, 0], [flowHeight-1, flowWidth-1]]])
#     tri = Delaunay(equiVertexArray)
#     triArray = tri.simplices
#     for y in range(h):
#         for x in range(w):
#             # NP
#             if y < h/3:
#                 rectImage = detRect(x, w, True)
#                 curX = x
#                 curY = y
#                 curVer = np.array([curY, curX, 1])
#                 flowVer = np.matmul(homographySet[0][rectImage], curVer)
#                 equiEndPoint = findRectFlowEndPoint(flowVer[0], flowVer[1],
#                                                     flowSet[0][rectImage], homographySet[0][rectImage])
#                 finalFlow[y][x] = np.array([equiEndPoint[0]-y, equiEndPoint[1]-x])
#             # SP
#             elif y >= h/1.5:
#                 rectImage = detRect(x, w, False)
#                 curX = x
#                 curY = y
#                 # It can be negative (on the left)
#                 # Should add w to x when doing warping
#                 # i.e. x = x + w
#                 if rectImage < 0:
#                     rectImage = rectImage + 5
#                     curX = curX + w
#                 curVer = np.array([curY, curX, 1])
#                 flowVer = np.matmul(homographySet[1][rectImage], curVer)
#                 equiEndPoint = findRectFlowEndPoint(flowVer[0], flowVer[1],
#                                                     flowSet[2][rectImage], homographySet[1][rectImage])
#                 finalFlow[y][x] = np.array([equiEndPoint[0]-y, equiEndPoint[1]-x])
#
#             # Mid
#             elif h/3 <= y < h/1.5:
#                 p = np.array([y, x])
#                 index = tri.find_simplex(p)
#                 # Find which image this pixel belongs to
#                 triImage = detTri(triArray[index])
#                 finalFlow[y][x] = getMidFinalFlow(h, w, p, flowSet, triImage, equiVertexArray, triOrientation)
#     return finalFlow


def getRectVertex(rectIdx, top, h, w):
    topMark = int(np.ceil(h/3)-1)
    botMark = int(np.ceil(h/1.5))
    if top:
        rectVertex = np.array([[0, int(np.floor(w*(0.1+0.2*rectIdx)))],
                               [topMark, int(np.floor(w*0.2*rectIdx))],
                               [topMark, int(np.floor(w*0.2*(rectIdx+1))-1)]])
    else:
        rectVertex = np.array([[botMark, int(np.floor(w*(0.1+0.2*rectIdx)))],
                               [botMark, int(np.floor(w*(0.3+0.2*rectIdx))-1)],
                               [h-1, int(np.floor(w*(rectIdx+1)*0.2)-1)]])
    return rectVertex


def detRect(x, w, top):
    if top:
        return int(np.ceil(5*(x+1)/w)-1)
    return int(np.ceil(5*(x+1)/w - 0.5)-1)


def detTri(indices):
    # Return array format:
    # flipped triangle: [top left, top right, bot, 1, index of images]s
    # non-flipped triangle: [top, bot left, bot right, 0, index of images]
    # Triangle 012
    if 1 in indices:
        if 0 in indices:
            return np.array([0, 1, 6, 1, 0])
        elif 2 in indices:
            return np.array([1, 2, 7, 1, 2])
        else:
            return np.array([1, 6, 7, 0, 1])
    # Triangle 345
    elif 8 in indices:
        if 7 in indices:
            return np.array([2, 7, 8, 0, 3])
        elif 9 in indices:
            return np.array([3, 8, 9, 0, 5])
        else:
            return np.array([2, 3, 8, 1, 4])
    # Triangle 678
    elif 4 in indices:
        if 3 in indices:
            return np.array([3, 4, 9, 1, 6])
        elif 5 in indices:
            return np.array([4, 5, 10, 1, 8])
        else:
            return np.array([4, 9, 10, 0, 7])
    elif 11 in indices:
        return np.array([0, 11, 6, 0, 9])
    # Triangle 9, 12 in indices
    return np.array([5, 10, 12, 0, 9])


# For finding triangles using bary
def findFlowValue(imgHeight, imgWidth, equiCurVer, equiVer0, equiVer1, equiVer2, flowVer0, flowVer1, flowVer2, flow):
    # Vertex in [x, y] format
    cart0, cart1, cart2 = equibaryPreProcess(imgHeight, imgWidth, equiVer0, equiVer1, equiVer2)
    # lambdaVal = newcart2bary(equiVer0, equiVer1, equiVer2, equiCurVer)
    lambdaVal = equiCoor2bary(imgHeight, imgWidth, cart0, cart1, cart2, equiCurVer)
    # Use lambda to get coordinate in flow matrix
    startPoint = lambdaVal[0] * flowVer0 + lambdaVal[1] * flowVer1 + lambdaVal[2] * flowVer2
    [h, w] = flow.shape[:2]
    # Get coefficient for bilinear interpolation
    # print(f'{equiCurVer}, {startPoint[0]}, {startPoint[1]}, {h}, {w}')
    biVal = bilinear_interpolation(startPoint[0], startPoint[1], h, w)
    # Get flow value after interpolation
    biFlow = biVal[4] * flow[int(biVal[0])][int(biVal[2])] + \
        biVal[5] * flow[int(biVal[0])][int(biVal[3])] + \
        biVal[6] * flow[int(biVal[1])][int(biVal[2])] + \
        biVal[7] * flow[int(biVal[1])][int(biVal[3])]
    # Get flow end point in triangle
    endPoint = startPoint + np.flip(biFlow)
    # Get endpoint lambda in triangle
    endLambda = newcart2bary(flowVer0, flowVer1, flowVer2, endPoint)
    # Convert endpoint back into equirectangular
    # endEquiVer = endLambda[0] * equiVer0 + endLambda[1] * equiVer1 + endLambda[2] * equiVer2
    endEquiVer = bary2equiCoor(imgHeight, imgWidth, cart0, cart1, cart2, endLambda)
    # Calculate flow in equirectangular
    equiFlow = np.flip(endEquiVer - equiCurVer)
    return equiFlow


# def findRectFlowEndPoint(r, c, flow, homography):
#     [h, w] = flow.shape
#     biVal = bilinear_interpolation(r, c, h, w)
#     startPoint = np.array([r, c])
#     # Get flow value at the current position
#     biFlow = biVal[4] * flow[int(biVal[0])][int(biVal[2])] + \
#         biVal[5] * flow[int(biVal[0])][int(biVal[3])] + \
#         biVal[6] * flow[int(biVal[1])][int(biVal[2])] + \
#         biVal[7] * flow[int(biVal[1])][int(biVal[3])]
#     # Get end point in flow
#     endPoint = startPoint + biFlow
#     # Get end point in equi
#     endEquiVer = np.matmul(np.linalg.inv(homography), np.array([endPoint[0], endPoint[1], 1]))
#     return endEquiVer


def getMidFinalFlow(imgHeight, imgWidth, p, flowSet, triImage, equiVertexArray, triOrientation):
    # flipped triangle: [top left, top right, bot, 1, index of images]s
    # non-flipped triangle: [top, bot left, bot right, 0, index of images]
    curFlow = findFlowValue(imgHeight, imgWidth, p, equiVertexArray[triImage[0]], equiVertexArray[triImage[1]],
                            equiVertexArray[triImage[2]], triOrientation[triImage[3]][0],
                            triOrientation[triImage[3]][1], triOrientation[triImage[3]][2],
                            flowSet[1][triImage[4]])
    return curFlow


def getMidVertexArray(h, w):
    # topThi = np.pi / 3
    # botThi = np.pi / 1.5
    topThi = np.pi / 2 - np.arctan(0.5)
    botThi = np.pi / 2 + np.arctan(0.5)
    intervalTheta = np.pi / 5
    equiVertexArray = np.zeros([13, 2])
    for vIdx in range(6):
        equiVertexArray[vIdx] = sphere2equi(h, w, np.array([vIdx*2*intervalTheta, topThi]))
    for vIdx in range(6, 11):
        equiVertexArray[vIdx] = sphere2equi(h, w, np.array([((vIdx-6)*2+1)*intervalTheta, botThi]))
    equiVertexArray[11] = sphere2equi(h, w, np.array([-1*intervalTheta, botThi]))
    equiVertexArray[12] = sphere2equi(h, w, np.array([11*intervalTheta, botThi]))
    return equiVertexArray


def getHomographySet(h, w, flowHeight, flowWidth):
    homographySet = np.zeros([2, 5, 3, 3])
    # Rectangle: top left, top right, bot right, bot left
    rectOrientation = np.array([[0, 0], [0, flowWidth-1], [flowHeight-1, flowWidth-1], [flowHeight-1, 0]])
    # For top tri/rect
    # Height from 0 to topMark
    topMark = int(np.ceil(h/3)-1)
    topVertexArray = np.zeros([5, 4, 2])
    for topIdx in range(5):
        # topVertexArray[topIdx] = np.array([[0, int(np.floor(w*0.2*topIdx))],
        #                                    [0, int(np.floor(w*0.2*(topIdx+1))-1)],
        #                                    [topMark, int(np.floor(w*0.2*(topIdx+1))-1)],
        #                                    [topMark, int(np.floor(w*0.2*topIdx))]])
        tmpArray = np.array([[0, int(np.floor(w*0.2*topIdx))],
                             [0, int(np.floor(w*0.2*(topIdx+1))-1)],
                             [topMark, int(np.floor(w*0.2*(topIdx+1))-1)],
                             [topMark, int(np.floor(w*0.2*topIdx))]])
        # Inverse for backward mapping
        homographySet[0][topIdx] = np.linalg.inv(cv2.findHomography(rectOrientation, tmpArray)[0])
    # For bot tri/rect
    # Height from botMark to h-1
    botMark = int(np.ceil(h/1.5))
    botVertexArray = np.zeros([5, 4, 2])
    for botIdx in range(5):
        # botVertexArray[botIdx] = np.array([[botMark, int(np.floor(w*(0.1+0.2*botIdx)))],
        #                                    [botMark, int(np.floor(w*(0.1+0.2*(botIdx+1)))-1)],
        #                                    [h-1, int(np.floor(w*(0.1+0.2*(botIdx+1)))-1)],
        #                                    [h-1, int(np.floor(w*(0.1+0.2*botIdx)))]])
        tmpArray = np.array([[botMark, int(np.floor(w*(0.1+0.2*botIdx)))],
                             [botMark, int(np.floor(w*(0.1+0.2*(botIdx+1)))-1)],
                             [h-1, int(np.floor(w*(0.1+0.2*(botIdx+1)))-1)],
                             [h-1, int(np.floor(w*(0.1+0.2*botIdx)))]])
        # Inverse for backward mapping
        homographySet[1][botIdx] = np.linalg.inv(cv2.findHomography(rectOrientation, tmpArray)[0])

    return homographySet

