import numpy as np
from numpy import cos, sin, arctan2, arccos

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
    return np.array([theta, thi])


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


def newcart2bary(p0, p1, p2, r):
    v0 = p1 - p0
    v1 = p2 - p0
    v2 = r - p0

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


def equibaryPreProcess(h, w, p0, p1, p2):
    sph0 = equi2sphere(h, w, p0[0], p0[1])
    sph1 = equi2sphere(h, w, p1[0], p1[1])
    sph2 = equi2sphere(h, w, p2[0], p2[1])
    cart0 = sphere2cart(sph0[0], sph0[1])
    cart1 = sphere2cart(sph1[0], sph1[1])
    cart2 = sphere2cart(sph2[0], sph2[1])
    return np.array([cart0, cart1, cart2])


def equiCoor2bary(h, w, cart0, cart1, cart2, p):
    sphP = equi2sphere(h, w, p[0], p[1])
    cartP = sphere2cart(sphP[0], sphP[1])
    d = pointIntersection(cart0, cart1, cart2, cartP)
    cartP = cartP + cartP * d
    ###################################
    # Need to convert is onto the triangle plane
    ##################################
    return newcart2bary(cart0, cart1, cart2, cartP)


def equiCoor2baryALLcart(cart0, cart1, cart2, cartP):
    d = pointIntersection(cart0, cart1, cart2, cartP)
    cartP = cartP + cartP * d
    ###################################
    # Need to convert is onto the triangle plane
    ##################################
    return newcart2bary(cart0, cart1, cart2, cartP)


def bary2equiCoor(h, w, cart0, cart1, cart2, endLambda):
    cartP = cart0 * endLambda[0] + cart1 * endLambda[1] + cart2 * endLambda[2]
    cartP = cartP / np.linalg.norm(cartP)
    sphP = cart2sphere(cartP[0], cartP[1], cartP[2])
    return sphere2equi(h, w, sphP)


def cart2sphere(x, y, z):
    # print(z)
    mag = np.linalg.norm([x, y, z])
    return np.array([arctan2(y/mag, x/mag), arccos(z/mag)])


def sphere2cart(theta, thi):
    return np.array([sin(thi)*cos(theta), sin(thi)*sin(theta), cos(thi)])


def pointIntersection(a, b, c, p):
    n = np.cross((b - a), (c - a))
    return np.dot(a - p, n) / np.dot(p, n)
