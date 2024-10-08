import cv2
import numpy as np
from equiprojection import sphere2equi
from scipy.spatial import Delaunay
from topimage import topImageFlow
from midimage import midImageFlow
from botimage import botImageFlow
from evalution import endPointError, flow_correction, angularError,\
    frameInterpolation, sphericalEndPointError, sphericalAugularError
import barymap
import flow_vis as fv
from flowio import readFlowFile, writeFlowFile

# vertex = np.array([[0, 0], [1, 0], [2, 0], [3, 0], [4, 0], [5, 0], [0.5, 1], [1.5, 1], [2.5, 1], [3.5, 1], [4.5, 1]])
# tri = Delaunay(vertex)
# triArray = tri.simplices
#
# # print(1 in triArray[1] and 7 in triArray[1] and 2 in triArray[1])
# p = np.array([1.5, 1.5])
# print(tri.find_simplex(p))


# w = 1000
# h = 480
#
# topMark = int(np.ceil(h / 3) - 1)
# topVertexArray = np.zeros([5, 4, 2])
# for topIdx in range(5):
#     topVertexArray[topIdx] = np.array([[0, int(np.floor(w * 0.2 * topIdx))],
#                                        [0, int(np.floor(w * 0.2 * (topIdx + 1)) - 1)],
#                                        [topMark, int(np.floor(w * 0.2 * (topIdx + 1)) - 1)],
#                                        [topMark, int(np.floor(w * 0.2 * topIdx))]])
#
# print (topVertexArray)
h = 480
w = 960
# topThi = np.pi / 3
# botThi = np.pi / 1.5
# intervalTheta = np.pi / 5
# equiVertexArray = np.zeros([13, 2])
# for vIdx in range(6):
#     equiVertexArray[vIdx] = sphere2equi(h, w, np.array([vIdx * 2 * intervalTheta, topThi]))
# for vIdx in range(6, 11):
#     equiVertexArray[vIdx] = sphere2equi(h, w, np.array([((vIdx - 6) * 2 + 1) * intervalTheta, botThi]))
#
# equiVertexArray[11] = sphere2equi(h, w, np.array([-1*intervalTheta, botThi]))
# equiVertexArray[12] = sphere2equi(h, w, np.array([11*intervalTheta, botThi]))
#
#
# # equiVertexArray = np.array([
# #     [np.floor(h/3), 0], [np.floor(h/3), np.floor(0.4*w)], [np.floor(h/3), np.floor(0.8*w)],
# #     [np.floor(h/3), np.floor(1.2*w)], [np.floor(h/3), np.floor(1.6*w)], [np.floor(h/3), np.floor(2*w)],
# #     [np.ceil(h/1.5)-1, np.floor(0.2*w)], [np.ceil(h/1.5)-1, np.floor(0.6*w)], [np.ceil(h/1.5)-1, w],
# #     [np.ceil(h/1.5)-1, np.floor(1.4*w)], [np.ceil(h/1.5)-1, np.floor(1.8*w)],
# #     [np.ceil(h/1.5)-1, np.floor(-0.2*w)], [np.ceil(h/1.5)-1, np.floor(2.2*w)]])
# print(equiVertexArray)


# botMark = int(np.ceil(h / 1.5))
# botVertexArray = np.zeros([5, 4, 2])
# for botIdx in range(5):
#     botVertexArray[botIdx] = np.array([[botMark, int(np.floor(w * (0.1 + 0.2 * botIdx)))],
#                                        [botMark, int(np.floor(w * (0.1 + 0.2 * (botIdx + 1))) - 1)],
#                                        [h - 1, int(np.floor(w * (0.1 + 0.2 * (botIdx + 1))) - 1)],
#                                        [h - 1, int(np.floor(w * (0.1 + 0.2 * botIdx)))]])
#
# print(botVertexArray)

readin1 = cv2.imread("GTFlow/hotel/0001_rgb.jpg")
readin2 = cv2.imread("GTFlow/hotel/0002_rgb.jpg")
#
# dirName = "testout/"
# topflow = topImageFlow(readin1, readin2, 200, 200, False)
# midflow = midImageFlow(readin1, readin2, 200, 200, True)
# botflow = botImageFlow(readin1, readin2, 200, 200, True)
#
# for idx, f in enumerate(topflow):
#     cv2.imwrite(f"{dirName}top{idx}.jpg", fv.flow_to_color(topflow[idx]))
# for idx, f in enumerate(midflow):
#     cv2.imwrite(f"{dirName}mid{idx}.jpg", fv.flow_to_color(midflow[idx]))
# for idx, f in enumerate(botflow):
#     cv2.imwrite(f"{dirName}bot{idx}.jpg", fv.flow_to_color(botflow[idx]))
#
filename = 'GTFlow/hotel/0001_opticalflow_forward.flo'
img1gray = cv2.cvtColor(readin1, cv2.COLOR_RGB2GRAY)
img2gray = cv2.cvtColor(readin2, cv2.COLOR_RGB2GRAY)

# triangleNumber = np.load('triangleNumber.npy')
# equiBiValSet = np.load('equiBiValSet.npy')
# equiStartPointSet = np.load('equiStartPointSet.npy')
# triCartSet = np.load('triCartSet.npy', allow_pickle=True)
# triangleFlowSet = np.load('flowSet.npy')

cubemapNumber = np.load('cubemapNumber.npy')
equiCubemapBiValSet = np.load('equiCubemapBiValSet.npy')
equiCubemapStartPointSet = np.load('equiCubemapStartPointSet.npy')
cubemapCartSet = np.load('cubemapCartSet.npy', allow_pickle=True)
cubemapFlowSet = np.load('cubemapFlowSet.npy')

disflow = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)

# fff = disflow.calc(img1gray, img2gray, None)
# fff = barymap.equiImg2EquiFlowBary(readin1, readin2)
# fff = barymap.equiImg2EquiFlowBary(img1gray, img2gray)
# fff = barymap.equiImg2EquiFlowBaryPreprocessed(img1gray, img2gray, triangleNumber, equiBiValSet,
#                                                equiStartPointSet, triCartSet, triangleFlowSet, disflow)
# fff = barymap.equiImg2EquiFlowCubemapPreprocessed(img1gray, img2gray, cubemapNumber, equiCubemapBiValSet,
#                                                   equiCubemapStartPointSet, cubemapCartSet, cubemapFlowSet, disflow)
fff = readFlowFile('icosahedron/hotel/0001.flo')
# correctedGT = flow_correction(readFlowFile(filename))
# calcf = flow_correction(fff)
# epe = endPointError(correctedGT, calcf)
# ae = angularError(correctedGT, calcf)
# sepe = sphericalEndPointError(correctedGT, calcf)
# sae = sphericalAugularError(correctedGT, calcf)
# print(epe)
# print(ae)
# print(sepe)
# print(sae)
# cv2.imwrite("testout/aaaaaa.jpg", fv.flow_to_color(fff))
out = frameInterpolation(fff, img2gray)
cv2.imwrite("testout/img2gray.jpg", out)
# writeFlowFile(fff, 'testout/test.flo')
