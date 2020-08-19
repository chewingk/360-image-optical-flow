import cv2
import numpy as np
from equiprojection import sphere2equi
from scipy.spatial import Delaunay
from topimage import topImageFlow
from midimage import midImageFlow
from botimage import botImageFlow
import flow_vis as fv

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

readin1 = cv2.imread("optical_flow_gt/0001_rgb.jpg")
readin2 = cv2.imread("optical_flow_gt/0002_rgb.jpg")

dirName = "testout/"
topflow = topImageFlow(readin1, readin2, 200, 200)
midflow = midImageFlow(readin1, readin2, 200, 200)
botflow = botImageFlow(readin1, readin2, 200, 200)

for idx, f in enumerate(topflow):
    cv2.imwrite(f"{dirName}top{idx}.jpg", fv.flow_to_color(topflow[idx]))
for idx, f in enumerate(midflow):
    cv2.imwrite(f"{dirName}mid{idx}.jpg", fv.flow_to_color(midflow[idx]))
for idx, f in enumerate(botflow):
    cv2.imwrite(f"{dirName}bot{idx}.jpg", fv.flow_to_color(botflow[idx]))

