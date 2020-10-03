import cv2
import numpy as np
from numpy import pi
import sys
import flow_vis as fv

# # feature_params = dict(maxCorners=100,
# #                       qualityLevel=0.3,
# #                       minDistance=7,
# #                       blockSize=7)
# # lk_params = dict(winSize=(15, 15),
# #                  maxLevel=2,
# #                  criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
# # color = np.random.randint(0, 255, (100, 3))
#
# img1 = cv2.imread(sys.argv[1])
# img1_gray = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
# img2 = cv2.imread(sys.argv[2])
# img2_gray = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
#
# mask = np.zeros_like(img1)
# # frame = img2
# #
# # p0 = cv2.goodFeaturesToTrack(img1_gray, mask=None, **feature_params)
# #
# # p1, st, err = cv2.calcOpticalFlowPyrLK(img1_gray, img2_gray, p0, None, **lk_params)
# #
# # good_new = p1[st == 1]
# # good_old = p0[st == 1]
# #
# # for i, (new, old) in enumerate(zip(good_new, good_old)):
# #     a, b = new.ravel()
# #     c, d = old.ravel()
# #     mask = cv2.line(mask, (a, b), (c, d), color[i].tolist(), 2)
# #     frame = cv2.circle(frame, (a, b), 5, color[i].tolist(), -1)
# #
# # outimg = cv2.add(frame, mask)
# #
# # cv2.imwrite('ttt.jpg', outimg)
#
# # flow = cv2.calcOpticalFlowFarneback(img1_gray, img2_gray, None, 0.5, 5, 11, 5, 5, 1.1, 0)
# #####
# aaa = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST)
# flow = aaa.calc(img1_gray, img2_gray, None)
# print(flow.shape)
#
# mask_rgb = fv.flow_to_color(flow)
#
# #####
# # mag, ang = cv2.cartToPolar(flow[:, :, 0], flow[:, :, 1])
# # mask[:, :, 0] = (ang * 180) / (np.pi * 2)
# # mask[:, :, 1] = 255
# # mask[:, :, 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
# # mask_rgb = cv2.cvtColor(mask, cv2.COLOR_HSV2BGR)
#
# # dense_flow = cv2.addWeighted(img1, 1, mask_rgb, 2, 0)
#
# # cv2.imwrite('ttt.jpg', dense_flow)
#
# cv2.imwrite('ooo2.jpg', mask_rgb)

# from barymap import pointInTriangle
from equiprojection import sphere2cart, equi2sphere, equiCoor2bary, newcart2bary, cart2sphere, sphere2equi
from visfunction import bilinear_interpolation

topLine = pi/2 - np.arctan(0.5)
botLine = pi/2 + np.arctan(0.5)

triOrientation = np.array([[[7, 109], [180, 9], [180, 209]],
                           [[8, 9], [8, 209], [181, 109]]])

flowHeight = 190
flowWidth = 220
imgHeight = 480
imgWidth = 960

hmax = 479
wmax = 959

a = np.array([0.2*pi, 0])
b = np.array([0.6*pi, 0])
c = np.array([pi, 0])
d = np.array([1.4*pi, 0])
e = np.array([1.8*pi, 0])
f = np.array([0, topLine])
g = np.array([0.4*pi, topLine])
h = np.array([0.8*pi, topLine])
i = np.array([1.2*pi, topLine])
j = np.array([1.6*pi, topLine])
k = np.array([2.0*pi, topLine])
l = np.array([0.2*pi, botLine])
m = np.array([0.6*pi, botLine])
n = np.array([pi, botLine])
o = np.array([1.4*pi, botLine])
p = np.array([1.8*pi, botLine])
q = np.array([0, pi])
r = np.array([0.4*pi, pi])
s = np.array([0.8*pi, pi])
t = np.array([1.2*pi, pi])
u = np.array([1.6*pi, pi])
v = np.array([2.0*pi, pi])

aCart = sphere2cart(a[0], a[1])
bCart = sphere2cart(b[0], b[1])
cCart = sphere2cart(c[0], c[1])
dCart = sphere2cart(d[0], d[1])
eCart = sphere2cart(e[0], e[1])
fCart = sphere2cart(f[0], f[1])
gCart = sphere2cart(g[0], g[1])
hCart = sphere2cart(h[0], h[1])
iCart = sphere2cart(i[0], i[1])
jCart = sphere2cart(j[0], j[1])
kCart = sphere2cart(k[0], k[1])
lCart = sphere2cart(l[0], l[1])
mCart = sphere2cart(m[0], m[1])
nCart = sphere2cart(n[0], n[1])
oCart = sphere2cart(o[0], o[1])
pCart = sphere2cart(p[0], p[1])
qCart = sphere2cart(q[0], q[1])
rCart = sphere2cart(r[0], r[1])
sCart = sphere2cart(s[0], s[1])
tCart = sphere2cart(t[0], t[1])
uCart = sphere2cart(u[0], u[1])
vCart = sphere2cart(v[0], v[1])

vertexp = list()
vertexp.append(np.array([aCart, fCart, gCart, False]))
vertexp.append(np.array([bCart, gCart, hCart, False]))
vertexp.append(np.array([cCart, hCart, iCart, False]))
vertexp.append(np.array([dCart, iCart, jCart, False]))
vertexp.append(np.array([eCart, jCart, kCart, False]))
vertexp.append(np.array([fCart, gCart, lCart, True]))
vertexp.append(np.array([gCart, lCart, mCart, False]))
vertexp.append(np.array([gCart, hCart, mCart, True]))
vertexp.append(np.array([hCart, mCart, nCart, False]))
vertexp.append(np.array([hCart, iCart, nCart, True]))
vertexp.append(np.array([iCart, nCart, oCart, False]))
vertexp.append(np.array([iCart, jCart, oCart, True]))
vertexp.append(np.array([jCart, oCart, pCart, False]))
vertexp.append(np.array([jCart, kCart, pCart, True]))
vertexp.append(np.array([kCart, pCart, lCart, False]))
vertexp.append(np.array([pCart, lCart, qCart, True]))
vertexp.append(np.array([lCart, mCart, rCart, True]))
vertexp.append(np.array([mCart, nCart, sCart, True]))
vertexp.append(np.array([nCart, oCart, tCart, True]))
vertexp.append(np.array([oCart, pCart, uCart, True]))


def getbiVal(equiP, cart0, cart1, cart2, flowVer0, flowVer1, flowVer2):
    lambdaVal = equiCoor2bary(imgHeight, imgWidth, cart0, cart1, cart2, equiP)

    startPoint = lambdaVal[0] * flowVer0 + lambdaVal[1] * flowVer1 + lambdaVal[2] * flowVer2
    # return bilinear_interpolation(startPoint[0], startPoint[1], flowHeight, flowWidth)
    return startPoint


def getFlowPreProcess(triArray, flip):
    # triArray 3 vals in cart
    flow = np.zeros([flowHeight, flowWidth, 8])
    if flip:
        triO = triOrientation[1]
    else:
        triO = triOrientation[0]
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
    if flip:
        np.rot90(flow, 2)
    return flow


#####################################################################
# Triangle Number output
# out = np.zeros([480, 960])
# for row in range(480):
#     for col in range(960):
#         point = equi2sphere(480, 960, row, col)
#         pointCart = sphere2cart(point[0], point[1])
#         if pointInTriangle(aCart, fCart, gCart, pointCart): #
#             out[row, col] = 1
#         elif pointInTriangle(bCart, gCart, hCart, pointCart): #
#             out[row, col] = 2
#         elif pointInTriangle(cCart, hCart, iCart, pointCart): #
#             out[row, col] = 3
#         elif pointInTriangle(dCart, iCart, jCart, pointCart): #
#             out[row, col] = 4
#         elif pointInTriangle(eCart, jCart, kCart, pointCart): #
#             out[row, col] = 5
#         elif pointInTriangle(fCart, gCart, lCart, pointCart): #
#             out[row, col] = 6
#         elif pointInTriangle(gCart, lCart, mCart, pointCart): #
#             out[row, col] = 7
#         elif pointInTriangle(gCart, hCart, mCart, pointCart): #
#             out[row, col] = 8
#         elif pointInTriangle(hCart, mCart, nCart, pointCart): #
#             out[row, col] = 9
#         elif pointInTriangle(hCart, iCart, nCart, pointCart): #
#             out[row, col] = 10
#         elif pointInTriangle(iCart, nCart, oCart, pointCart): #
#             out[row, col] = 11
#         elif pointInTriangle(iCart, jCart, oCart, pointCart): #
#             out[row, col] = 12
#         elif pointInTriangle(jCart, oCart, pCart, pointCart): #
#             out[row, col] = 13
#         elif pointInTriangle(jCart, kCart, pCart, pointCart): #
#             out[row, col] = 14
#         elif pointInTriangle(kCart, pCart, lCart, pointCart): #
#             out[row, col] = 15
#         elif pointInTriangle(pCart, lCart, qCart, pointCart): #
#             out[row, col] = 16
#         elif pointInTriangle(lCart, mCart, rCart, pointCart): #
#             out[row, col] = 17
#         elif pointInTriangle(mCart, nCart, sCart, pointCart): #
#             out[row, col] = 18
#         elif pointInTriangle(nCart, oCart, tCart, pointCart): #
#             out[row, col] = 19
#         elif pointInTriangle(oCart, pCart, uCart, pointCart): #
#             out[row, col] = 20
#         # else:
#         #     print(row, col)
#
#
# for row in range(480):
#     for col in range(960):
#         if out[row, col] == 0:
#             tempList = []
#             if row > 0:
#                 if col > 0:
#                     if out[row-1, col-1] != 0:
#                         tempList.append(out[row-1, col-1])
#                 if col < wmax:
#                     if out[row-1, col+1] != 0:
#                         tempList.append(out[row-1, col+1])
#                 if out[row-1, col] != 0:
#                     tempList.append(out[row-1, col])
#             if row < hmax:
#                 if col > 0:
#                     if out[row+1, col-1] != 0:
#                         tempList.append(out[row+1, col-1])
#                 if col < wmax:
#                     if out[row+1, col+1] != 0:
#                         tempList.append(out[row+1, col+1])
#                 if out[row+1, col] != 0:
#                     tempList.append(out[row+1, col])
#             if col > 0:
#                 if out[row, col-1] != 0:
#                     tempList.append(out[row, col-1])
#             if col < wmax:
#                 if out[row, col+1] != 0:
#                     tempList.append(out[row, col+1])
#             out[row, col] = max(set(tempList), key=tempList.count)
#
#
# for row in range(480):
#     for col in range(960):
#         if out[row, col] == 0:
#             print(row, col)
#
# out = out - 1
#
# # cv2.imwrite('testout/pointintri.jpg', out)
# np.save('triangleNumber.npy', out)
###############################################################
###############################################################
# FlowSet output

# flowSet = np.zeros([20, flowHeight, flowWidth, 8])
#
# for idx, ver in enumerate(vertexp):
#     flowSet[idx] = getFlowPreProcess(ver[0:3], ver[3])
#
# np.save('flowSet.npy', flowSet)
#############################################################
###########################################################
# # Equi BiVal
# triangleNumber = np.load('triangleNumber.npy')
# # equiBiValSet = np.zeros([imgHeight, imgWidth, 8])
# equiStartPointSet = np.zeros([imgHeight, imgWidth, 2])
#
# for row in range(imgHeight):
#     for col in range(imgWidth):
#         equiCurVer = np.array([row, col])
#         index = int(triangleNumber[row, col])
#         triInfo = vertexp[index]
#         if triInfo[3]:
#             ori = triOrientation[1]
#         else:
#             ori = triOrientation[0]
#         # equiBiValSet[row, col] = getbiVal(equiCurVer, triInfo[0], triInfo[1], triInfo[2],
#         #                                   ori[0], ori[1], ori[2])
#         equiStartPointSet[row, col] = getbiVal(equiCurVer, triInfo[0], triInfo[1], triInfo[2],
#                                                ori[0], ori[1], ori[2])
#
# # np.save('equiBiValSet.npy', equiBiValSet)
# np.save('equiStartPointSet.npy', equiStartPointSet)

###############################################################
###############################################################
triCartSet = np.array(vertexp)
np.save('triCartSet.npy', triCartSet)