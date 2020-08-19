import numpy as np


class Icosaherdron:
    vertex = []
    faces = []

    def __init__(self):
        for v in [SphereVertices(0, 0), SphereVertices(0, np.pi/3), SphereVertices(2*np.pi/5, np.pi/3),
                  SphereVertices(4*np.pi/5, np.pi/3), SphereVertices(6*np.pi/5, np.pi/3), SphereVertices(8*np.pi/5, np.pi/3),
                  SphereVertices(np.pi/5, 2*np.pi/3), SphereVertices(3*np.pi/5, 2*np.pi/3), SphereVertices(np.pi, 2*np.pi/3),
                  SphereVertices(7*np.pi/5, 2*np.pi/3), SphereVertices(9*np.pi/5, 2*np.pi/3), SphereVertices(0, np.pi)]:
            self.vertex.append(v)


class Triangle:

    def __init__(self, ver1, ver2, ver3):
        self.ver1 = ver1
        self.ver2 = ver2
        self.ver3 = ver3

#
# class DownTriangle:
#
#     def __init__(self, left, right, bot):
#         self.left = left
#         self.right = right
#         self.bot = bot


class BaryVertices:
    def __init__(self, b1, b2, b3):
        self.b1 = b1
        self.b2 = b2
        self.b3 = b3


class CartVertices:
    def __init__(self, x, y):
        self.x = x
        self.y = y


class SphereVertices:
    def __init__(self, theta, thi):
        self.theta = theta
        self.thi = thi


# class Face:
