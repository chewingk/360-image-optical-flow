import numpy as np
import cv2
from icosaherdron import Triangle, BaryVertices, CartVertices, SphereVertices


# North pole as (0,0) for sphere
# Top left as (0,0) for equi
def sphere2equi(h, w, vertex):
    c = w * vertex[0] / (2 * np.pi)
    r = h * vertex[1] / np.pi
    return np.array([r, c])


def equi2sphere(h, w, r, c):
    theta = 2 * np.pi * c / w
    thi = np.pi * r / h
    return [theta, thi]


def cart2bary(p0, p1, p2, r):

    # print([p0, p1, p2, r])
    y2my3 = p1[1] - p2[1]
    x3mx2 = p2[0] - p1[0]
    x1mx3 = p0[0] - p2[0]
    y1my3 = p0[1] - p2[1]
    xmx3 = r[0] - p2[0]
    ymy3 = r[1] - p2[1]

    lambda1 = (y2my3 * xmx3  + x3mx2 * ymy3) / (y2my3 * x1mx3 + x3mx2 * y1my3)
    lambda2 = (x1mx3 * ymy3 - y1my3 * xmx3) / (y2my3 * x1mx3 + x3mx2 * y1my3)
    lambda3 = 1 - lambda1 - lambda2

    return np.array([lambda1, lambda2, lambda3])


def bary2cart(triangle, bary):
    x = triangle.ver1.x * bary.b1 + triangle.ver2.x * bary.b2 + triangle.ver3.x * bary.b3
    y = triangle.ver1.y * bary.b1 + triangle.ver2.y * bary.b2 + triangle.ver3.y * bary.b3
    return CartVertices(x, y)


def newcart2bary(p1, p2, p3, r):
    v0 = p2 - p1
    v1 = p3 - p1
    v2 = r - p1

    d00 = np.dot(v0, v0)
    d01 = np.dot(v0, v1)
    d11 = np.dot(v1, v1)
    d20 = np.dot(v2, v0)
    d21 = np.dot(v2, v1)
    denom = d00 * d11 - d01 * d01

    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1.0 - v - w

    return np.array([u, v, w])
