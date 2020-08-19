import cv2
import numpy as np
from barymap import getImageTriangle

# I have a equi image
# equi = cv2.imread('')
# h, w = equi.shape[:2]
h = 200
w = 200

# readin1 = cv2.imread('360images/frame145.jpg')
# readin2 = cv2.imread('360images/frame147.jpg')
readin1 = cv2.imread("optical_flow_gt/0001_rgb.jpg")
readin2 = cv2.imread("optical_flow_gt/0002_rgb.jpg")
# 121067----------------------------------- 1 Flip
# vertexp = np.array([[0.4*np.pi, np.pi/3],
#                     [0, np.pi/3],
#                     [0.2*np.pi, np.pi/1.5],
#                     [0.6*np.pi, np.pi/1.5],
#                     [-0.2*np.pi, np.pi/1.5]])
# 12367---------------------------------- 2
# vertexp = np.array([[0.2*np.pi, np.pi/1.5],
#                     [0.6*np.pi, np.pi/1.5],
#                     [0.4*np.pi, np.pi/3],
#                     [0, np.pi/3],
#                     [0.8*np.pi, np.pi/3]])
# 23678---------------------------------- 3 Flip
# vertexp = np.array([[0.8*np.pi, np.pi/3],
#                     [0.4*np.pi, np.pi/3],
#                     [0.6*np.pi, np.pi/1.5],
#                     [np.pi, np.pi/1.5],
#                     [0.2*np.pi, np.pi/1.5]])
# 78234---------------------------------- 4
# vertexp = np.array([[0.6*np.pi, np.pi/1.5],
#                     [np.pi, np.pi/1.5],
#                     [0.8*np.pi, np.pi/3],
#                     [0.4*np.pi, np.pi/3],
#                     [1.2*np.pi, np.pi/3]])
# 34789---------------------------------- 5 Flip
# vertexp = np.array([[1.2*np.pi, np.pi/3],
#                     [0.8*np.pi, np.pi/3],
#                     [np.pi, np.pi/1.5],
#                     [1.4*np.pi, np.pi/1.5],
#                     [0.6*np.pi, np.pi/1.5]])
# 34589----------------------------------- 6
# vertexp = np.array([[np.pi, np.pi/1.5],
#                     [1.4*np.pi, np.pi/1.5],
#                     [1.2*np.pi, np.pi/3],
#                     [0.8*np.pi, np.pi/3],
#                     [1.6*np.pi, np.pi/3]])
# 458910----------------------------------- 7 Flip
# vertexp = np.array([[1.6*np.pi, np.pi/3],
#                     [1.2*np.pi, np.pi/3],
#                     [1.4*np.pi, np.pi/1.5],
#                     [1.8*np.pi, np.pi/1.5],
#                     [np.pi, np.pi/1.5]])
# 451910----------------------------------- 8
# vertexp = np.array([[1.4*np.pi, np.pi/1.5],
#                     [1.8*np.pi, np.pi/1.5],
#                     [1.6*np.pi, np.pi/3],
#                     [1.2*np.pi, np.pi/3],
#                     [2*np.pi, np.pi/3]])
# 519106----------------------------------- 9 Flip
# vertexp = np.array([[2*np.pi, np.pi/3],
#                     [1.6*np.pi, np.pi/3],
#                     [1.8*np.pi, np.pi/1.5],
#                     [2.2*np.pi, np.pi/1.5],
#                     [1.4*np.pi, np.pi/1.5]])
# 106512----------------------------------- 10
# vertexp = np.array([[1.8*np.pi, np.pi/1.5],
#                     [2.2*np.pi, np.pi/1.5],
#                     [2*np.pi, np.pi/3],
#                     [1.6*np.pi, np.pi/3],
#                     [2.4*np.pi, np.pi/3]])


# 23NP14 XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
# alpha = (np.sin(0.3*np.pi) / np.sin(0.4*np.pi)) ** 2
# alpha = 1
# beta = 1 - alpha
# point1 = [0, np.pi/3]
# point2 = [0.2*np.pi, 0]
# point3 = [np.pi, 0]
# point4 = [1.2*np.pi, np.pi/3]
# fused1 = np.multiply(alpha, point2) + np.multiply(beta, point1)
# fused2 = np.multiply(alpha, point3) + np.multiply(beta, point4)
# vertexp = np.array([[0.4*np.pi, np.pi/3],
#                     [0.8*np.pi, np.pi/3],
#                     [0.6*np.pi, 0],
#                     fused1,
#                     fused2])
# 87SP96 XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX flip
# vertexp = np.array([[np.pi, np.pi/1.5],
#                     [0.6*np.pi, np.pi/1.5],
#                     [0.8*np.pi, np.pi],
#                     [1.2*np.pi, np.pi],
#                     [0.4*np.pi, np.pi]])

# 610SP XXXXXXXXXXXXXXXXXXXXXXXXXXX flip
# vertexp = np.array([[2.2*np.pi, np.pi/1.5],
#                     [1.8*np.pi, np.pi/1.5],
#                     [2*np.pi, np.pi],
#                     [2.4*np.pi, np.pi],
#                     [1.6*np.pi, np.pi]])

# 12NP XXXXXXXXXXXXXXXXXXXXXXXXXXXXX
vertexp = np.array([[0, np.pi/3],
                    [0.4*np.pi, np.pi/3],
                    [0.2*np.pi, 0],
                    [-0.2*np.pi, 0],
                    [0.6*np.pi, 0]])

# For poles
# out1 = lalaa(vertexp, h, w, readin1, pole=True, tri0theta=0.2*np.pi, tri2theta=np.pi)
# out2 = lalaa(vertexp, h, w, readin2, pole=True, tri0theta=0.2*np.pi, tri2theta=np.pi)
# For ^ triangle
out1 = getImageTriangle(vertexp, h, w, readin1)
out2 = getImageTriangle(vertexp, h, w, readin2)
# For V triangle (flip)
# out1 = getImageTriangle(vertexp, h, w, readin1, flip=True)
# out2 = getImageTriangle(vertexp, h, w, readin2, flip=True)
# cv2.imwrite('lalaa3.jpg', out1)
# cv2.imwrite('lalaa4.jpg', out2)
cv2.imwrite('lalaa5.jpg', out1)
cv2.imwrite('lalaa6.jpg', out2)
