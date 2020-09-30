import cv2
import numpy as np
from gettriangleimage import getImageTriangle, getImageBary

# I have a equi image
# equi = cv2.imread('')
# h, w = equi.shape[:2]
# h = 200
# w = 200

topLine = np.pi / 2 - np.arctan(0.5)
botLine = np.pi / 2 + np.arctan(0.5)
# readin1 = cv2.imread("GTFlow/hotel/0001_rgb.jpg")
# readin2 = cv2.imread("GTFlow/hotel/0002_rgb.jpg")
readin1 = cv2.imread("grid.png")
# 121067----------------------------------- 1 Flip
# vertexp = np.array([[0.4*np.pi, topLine],
#                     [0, topLine],
#                     [0.2*np.pi, botLine],
#                     [0.6*np.pi, botLine],
#                     [-0.2*np.pi, botLine]])
# 12367---------------------------------- 2
# vertexp = np.array([[0.2*np.pi, botLine],
#                     [0.6*np.pi, botLine],
#                     [0.4*np.pi, topLine],
#                     [0, topLine],
#                     [0.8*np.pi, topLine]])
# 23678---------------------------------- 3 Flip
# vertexp = np.array([[0.8*np.pi, topLine],
#                     [0.4*np.pi, topLine],
#                     [0.6*np.pi, botLine],
#                     [np.pi, botLine],
#                     [0.2*np.pi, botLine]])
# 78234---------------------------------- 4
# vertexp = np.array([[0.6*np.pi, botLine],
#                     [np.pi, botLine],
#                     [0.8*np.pi, topLine],
#                     [0.4*np.pi, topLine],
#                     [1.2*np.pi, topLine]])
# 34789---------------------------------- 5 Flip
# vertexp = np.array([[1.2*np.pi, topLine],
#                     [0.8*np.pi, topLine],
#                     [np.pi, botLine],
#                     [1.4*np.pi, botLine],
#                     [0.6*np.pi, botLine]])
# 34589----------------------------------- 6
# vertexp = np.array([[np.pi, botLine],
#                     [1.4*np.pi, botLine],
#                     [1.2*np.pi, topLine],
#                     [0.8*np.pi, topLine],
#                     [1.6*np.pi, topLine]])
# 458910----------------------------------- 7 Flip
# vertexp = np.array([[1.6*np.pi, topLine],
#                     [1.2*np.pi, topLine],
#                     [1.4*np.pi, botLine],
#                     [1.8*np.pi, botLine],
#                     [np.pi, botLine]])
# 451910----------------------------------- 8
# vertexp = np.array([[1.4*np.pi, botLine],
#                     [1.8*np.pi, botLine],
#                     [1.6*np.pi, topLine],
#                     [1.2*np.pi, topLine],
#                     [2*np.pi, topLine]])
# 519106----------------------------------- 9 Flip
# vertexp = np.array([[2*np.pi, topLine],
#                     [1.6*np.pi, topLine],
#                     [1.8*np.pi, botLine],
#                     [2.2*np.pi, botLine],
#                     [1.4*np.pi, botLine]])
# 106512----------------------------------- 10
# vertexp = np.array([[1.8*np.pi, botLine],
#                     [2.2*np.pi, botLine],
#                     [2*np.pi, topLine],
#                     [1.6*np.pi, topLine],
#                     [2.4*np.pi, topLine]])


# 23NP14 XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
# alpha = (np.sin(0.3*np.pi) / np.sin(0.4*np.pi)) ** 2
# alpha = 1
# beta = 1 - alpha
# point1 = [0, topLine]
# point2 = [0.2*np.pi, 0]
# point3 = [np.pi, 0]
# point4 = [1.2*np.pi, topLine]
# fused1 = np.multiply(alpha, point2) + np.multiply(beta, point1)
# fused2 = np.multiply(alpha, point3) + np.multiply(beta, point4)
# vertexp = np.array([[0.4*np.pi, topLine],
#                     [0.8*np.pi, topLine],
#                     [0.6*np.pi, 0],
#                     fused1,
#                     fused2])
# 87SP96 XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX flip
# vertexp = np.array([[np.pi, botLine],
#                     [0.6*np.pi, botLine],
#                     [0.8*np.pi, np.pi],
#                     [1.2*np.pi, np.pi],
#                     [0.4*np.pi, np.pi]])

# 610SP XXXXXXXXXXXXXXXXXXXXXXXXXXX flip
# vertexp = np.array([[2.2*np.pi, botLine],
#                     [1.8*np.pi, botLine],
#                     [2*np.pi, np.pi],
#                     [2.4*np.pi, np.pi],
#                     [1.6*np.pi, np.pi]])

# 12NP XXXXXXXXXXXXXXXXXXXXXXXXXXXXX
vertexp = np.array([[0, topLine],
                    [0.4*np.pi, topLine],
                    [0.2*np.pi, 0],
                    [-0.2*np.pi, 0],
                    [0.6*np.pi, 0]])

# Mid flip
# baryexp = np.array([[0.4*np.pi, topLine],
#                     [0.8*np.pi, topLine],
#                     [0.6*np.pi, botLine]])
# Bot
# baryexp = np.array([[0.6*np.pi, botLine],
#                     [np.pi, botLine],
#                     [0.8*np.pi, np.pi]])
# Top
# baryexp = np.array([[0.6*np.pi, 0],
#                     [0.4*np.pi, topLine],
#                     [0.8*np.pi, topLine]])
# baryexp = np.array([[0.8*np.pi, topLine],
#                     [0.6*np.pi, botLine],
#                     [np.pi, botLine]])
baryexp = np.array([[0.4*np.pi, topLine],
                    [0.8*np.pi, topLine],
                    [0.6*np.pi, botLine]])


img1gray = cv2.cvtColor(readin1, cv2.COLOR_RGB2GRAY)
# img2gray = cv2.cvtColor(readin2, cv2.COLOR_RGB2GRAY)

# For bary triangles
# out1 = getImageBary(baryexp, readin1, flip=False)
# out2 = getImageBary(baryexp, readin2, flip=False)
out1 = getImageBary(baryexp, img1gray, flip=True)
# out2 = getImageBary(baryexp, img2gray, flip=False)

# For poles
# out1 = lalaa(vertexp, h, w, readin1, pole=True, tri0theta=0.2*np.pi, tri2theta=np.pi)
# out2 = lalaa(vertexp, h, w, readin2, pole=True, tri0theta=0.2*np.pi, tri2theta=np.pi)
# For ^ triangle
# out1 = getImageTriangle(vertexp, h, w, readin1)
# out2 = getImageTriangle(vertexp, h, w, readin2)
# For V triangle (flip)
# out1 = getImageTriangle(vertexp, h, w, readin1, flip=True)
# out2 = getImageTriangle(vertexp, h, w, readin2, flip=True)
# cv2.imwrite('lalaa3.jpg', out1)
# cv2.imwrite('lalaa4.jpg', out2)
cv2.imwrite('lalaa5bary.jpg', out1)
# cv2.imwrite('lalaa6bary.jpg', out2)
