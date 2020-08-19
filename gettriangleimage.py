import cv2
import numpy as np
from scipy.spatial import Delaunay
from equiprojection import sphere2equi, cart2bary
from visfunction import bilinear_interpolation


# p: 5 points, each with its theta and thi, np array
# w: width of image
# h: height of image
def getImageTriangle(p, h, w, equiImg, flip=False, pole=False, tri0theta=None, tri2theta=None):
    equiDimension = equiImg.shape
    # !!!!!!!! IMPORTANT !!!!!!!
    # Delaunay uses x,y system
    # Image uses r,c system so they are flipped
    vertex = np.array([[0.5*w, h], [1.5*w, h], [w, 0], [0, 0], [2*w, 0]])
    tri = Delaunay(vertex)
    triArray = tri.simplices
    # Assuming rgb image
    out = np.zeros([h, 2*w, 3])
    for y in range(h):
        for x in range(2*w):
            if not inRect(x, w):
                continue
            else:
                curVer = np.array([x, y])
                index = tri.find_simplex(curVer)
                corner0 = triArray[index][0]
                corner1 = triArray[index][1]
                corner2 = triArray[index][2]
                lambdaValue = cart2bary(vertex[corner0], vertex[corner1],
                                        vertex[corner2], curVer)
                # p[corner] => point
                # p[corner][0] => theta & p[corner][1] => thi
                triVer0 = p[corner0]
                triVer1 = p[corner1]
                triVer2 = p[corner2]
                if pole:
                    if index == 0:
                        triVer0[0] = tri0theta
                    elif index == 2:
                        triVer2[0] = tri2theta
                theta = triVer0[0] * lambdaValue[0] + triVer1[0] * lambdaValue[1] + triVer2[0] * lambdaValue[2]
                thi = triVer0[1] * lambdaValue[0] + triVer1[1] * lambdaValue[1] + triVer2[1] * lambdaValue[2]
                while theta < 0:
                    theta = theta + 2*np.pi
                while thi > 2*np.pi:
                    theta = theta - 2*np.pi
                # while thi < 0:
                #     thi = thi +
                # print(f"Theta: {theta}; Thi: {thi}")
                # For each pixel convert its theta and thi into r and c in equi
                equiCoor = sphere2equi(equiDimension[0], equiDimension[1], np.array([theta, thi]))
                bilinearVal = bilinear_interpolation(equiCoor[0], equiCoor[1], equiDimension[0], equiDimension[1])
                # Assuming rgb image
                for rgb in range(3):
                    out[y][x][rgb] = equiImg[int(bilinearVal[0])][int(bilinearVal[2])][rgb] * bilinearVal[4] + \
                        equiImg[int(bilinearVal[0])][int(bilinearVal[3])][rgb] * bilinearVal[5] + \
                        equiImg[int(bilinearVal[1])][int(bilinearVal[2])][rgb] * bilinearVal[6] + \
                        equiImg[int(bilinearVal[1])][int(bilinearVal[3])][rgb] * bilinearVal[7]

    if flip:
        return np.rot90(out[:, int(0.5*w):int(1.5*w)], 2)
    return out[:, int(0.5*w):int(1.5*w)]


def inRect(x, w):
    if 0.5*w <= x < 1.5*w:
        return True
    return False
