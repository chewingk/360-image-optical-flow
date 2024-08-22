import cv2
import numpy as np
from numpy import pi, sqrt

from barymap import pointInTriangle
from equiprojection import sphere2cart, equi2sphere, equiCoor2bary, newcart2bary, cart2sphere, sphere2equi
from visfunction import bilinear_interpolation

topLine = pi/2 - 1/sqrt(3)
botLine = pi/2 + 1/sqrt(3)

flowHeight = 300
flowWidth = 300
imgHeight = 480
imgWidth = 960

hmax = imgHeight - 1
wmax = imgWidth - 1

a = np.array([0, topLine])
b = np.array([pi/2, topLine])
c = np.array([pi, topLine])
d = np.array([3*pi/2, topLine])
e = np.array([0, botLine])
f = np.array([pi/2, botLine])
g = np.array([pi, botLine])
h = np.array([3*pi/2, botLine])

aCart = sphere2cart(a[0], a[1])
bCart = sphere2cart(b[0], b[1])
cCart = sphere2cart(c[0], c[1])
dCart = sphere2cart(d[0], d[1])
eCart = sphere2cart(e[0], e[1])
fCart = sphere2cart(f[0], f[1])
gCart = sphere2cart(g[0], g[1])
hCart = sphere2cart(h[0], h[1])

vertexp = list()
vertexp.append(np.array([cCart, aCart, bCart]))
vertexp.append(np.array([bCart, eCart, fCart]))
vertexp.append(np.array([cCart, fCart, gCart]))
vertexp.append(np.array([dCart, gCart, hCart]))
vertexp.append(np.array([aCart, hCart, eCart]))
vertexp.append(np.array([fCart, hCart, gCart]))


def getbiVal(equiP, cart0, cart1, cart2, flowVer0, flowVer1, flowVer2):
    lambdaVal = equiCoor2bary(imgHeight, imgWidth, cart0, cart1, cart2, equiP)

    startPoint = lambdaVal[0] * flowVer0 + lambdaVal[1] * flowVer1 + lambdaVal[2] * flowVer2
    # return bilinear_interpolation(startPoint[0], startPoint[1], flowHeight, flowWidth)
    return startPoint, bilinear_interpolation(startPoint[0], startPoint[1], flowHeight, flowWidth)


def getFlowPreProcess(triArray):
    # triArray 3 vals in cart
    flow = np.zeros([flowHeight, flowWidth, 8])
    triO = np.array([[29, 269], [269, 29], [269, 269]])
    # if flip:
    #     triO = triOrientation[1]
    # else:
    #     triO = triOrientation[0]
    cart0, cart1, cart2 = triArray
    for row in range(flowHeight):
        for col in range(flowWidth):
            curVer = np.array([row, col])
            curLambda = newcart2bary(triO[0], triO[1], triO[2], curVer)
            curCart = cart0 * curLambda[0] + cart1 * curLambda[1] + cart2 * curLambda[2]
            curCart = curCart / np.linalg.norm(curCart)

            curTheta, curThi = cart2sphere(curCart[0], curCart[1], curCart[2])
            if curThi < 0:
                curThi = -1 * curThi
                curTheta = curTheta + pi
            elif curThi > pi:
                curThi = 2 * pi - curThi
                curTheta = curTheta + pi
            while curTheta < 0:
                curTheta = curTheta + 2 * pi
            while curTheta > 2 * pi:
                curTheta = curTheta - 2 * pi

            equiCoor = sphere2equi(imgHeight, imgWidth, np.array([curTheta, curThi]))
            flow[row, col] = bilinear_interpolation(equiCoor[0], equiCoor[1], imgHeight, imgWidth)
    # if flip:
    #     np.rot90(flow, 2)
    # np.rot90(flow, 2)
    return flow
# #####################################################################
# # Cubemap Number output
out = np.zeros([480, 960])
for row in range(480):
    for col in range(960):
        point = equi2sphere(480, 960, row, col)
        pointCart = sphere2cart(point[0], point[1])
        if pointInTriangle(aCart, bCart, cCart, pointCart) or pointInTriangle(aCart, cCart, dCart, pointCart):
            out[row, col] = 1
        elif pointInTriangle(aCart, bCart, eCart, pointCart) or pointInTriangle(bCart, eCart, fCart, pointCart):
            out[row, col] = 2
        elif pointInTriangle(bCart, cCart, fCart, pointCart) or pointInTriangle(cCart, fCart, gCart, pointCart):
            out[row, col] = 3
        elif pointInTriangle(cCart, dCart, gCart, pointCart) or pointInTriangle(dCart, gCart, hCart, pointCart):
            out[row, col] = 4
        elif pointInTriangle(dCart, aCart, hCart, pointCart) or pointInTriangle(aCart, hCart, eCart, pointCart):
            out[row, col] = 5
        elif pointInTriangle(eCart, fCart, hCart, pointCart) or pointInTriangle(fCart, hCart, gCart, pointCart):
            out[row, col] = 6
        # else:
        #     print(row, col)

for row in range(480):
    for col in range(960):
        if out[row, col] == 0:
            tempList = []
            if row > 0:
                if col > 0:
                    if out[row-1, col-1] != 0:
                        tempList.append(out[row-1, col-1])
                if col < wmax:
                    if out[row-1, col+1] != 0:
                        tempList.append(out[row-1, col+1])
                if out[row-1, col] != 0:
                    tempList.append(out[row-1, col])
            if row < hmax:
                if col > 0:
                    if out[row+1, col-1] != 0:
                        tempList.append(out[row+1, col-1])
                if col < wmax:
                    if out[row+1, col+1] != 0:
                        tempList.append(out[row+1, col+1])
                if out[row+1, col] != 0:
                    tempList.append(out[row+1, col])
            if col > 0:
                if out[row, col-1] != 0:
                    tempList.append(out[row, col-1])
            if col < wmax:
                if out[row, col+1] != 0:
                    tempList.append(out[row, col+1])
            out[row, col] = max(set(tempList), key=tempList.count)

# for row in range(480):
#     for col in range(960):
#         if out[row, col] == 0:
#             print(row, col)

# out = out * 40
# cv2.imwrite('testout/cubemapClassificationFULL.jpg', out)
out = out - 1
np.save('cubemapNumber.npy', out)
###############################################################
###############################################################
# FlowSet output
flowSet = np.zeros([6, flowHeight, flowWidth, 8])

for idx, ver in enumerate(vertexp):
    flowSet[idx] = getFlowPreProcess(ver[0:3])

np.save('cubemapFlowSet.npy', flowSet)
#############################################################
###########################################################
# Equi BiVal
cubemapNumber = np.load('cubemapNumber.npy')
equiBiValSet = np.zeros([imgHeight, imgWidth, 8])
equiStartPointSet = np.zeros([imgHeight, imgWidth, 2])

for row in range(imgHeight):
    for col in range(imgWidth):
        equiCurVer = np.array([row, col])
        index = int(cubemapNumber[row, col])
        triInfo = vertexp[index]
        ori = np.array([[29, 269], [269, 29], [269, 269]])
        # if triInfo[3]:
        #     ori = triOrientation[1]
        # else:
        #     ori = triOrientation[0]
        # equiBiValSet[row, col] = getbiVal(equiCurVer, triInfo[0], triInfo[1], triInfo[2],
        #                                   ori[0], ori[1], ori[2])
        equiStartPointSet[row, col], equiBiValSet[row, col] = getbiVal(equiCurVer, triInfo[0], triInfo[1],
                                                                       triInfo[2], ori[0], ori[1], ori[2])

np.save('equiCubemapBiValSet.npy', equiBiValSet)
np.save('equiCubemapStartPointSet.npy', equiStartPointSet)

##############################################################
##############################################################
cubemapCartSet = np.array(vertexp)
np.save('cubemapCartSet.npy', cubemapCartSet)
