import cv2
import numpy
import sys
import numpy as np

from numpy import pi
from math import pi, sin, cos, tan, sqrt
from PIL import Image

from equiprojection import equi2sphere
from visfunction import bilinear_interpolation

# Convert using an inverse transformation
def generate_mapping_data(image_width):
    in_size = [image_width, int(image_width * 3 / 4)]
    edge = in_size[0] / 4  # The length of each edge in pixels

    # Create our numpy arrays
    out_pix = numpy.zeros((in_size[1], in_size[0], 2), dtype="f4")
    xyz = numpy.zeros((int(in_size[1] * in_size[0] / 2), 3), dtype="f4")
    vals = numpy.zeros((int(in_size[1] * in_size[0] / 2), 3), dtype="i4")

    # Much faster to use an arange when we assign to to vals
    start, end = 0, 0
    rng_1 = numpy.arange(0, edge * 3)
    rng_2 = numpy.arange(edge, edge * 2)
    for i in range(in_size[0]):
        # 0: back
        # 1: left
        # 2: front
        # 3: right
        face = int(i / edge)
        rng = rng_1 if face == 2 else rng_2

        end += len(rng)
        vals[start:end, 0] = rng
        vals[start:end, 1] = i
        vals[start:end, 2] = face
        start = end

    # Top/bottom are special conditions
    j, i, face = vals.T
    face[j < edge] = 4  # top
    face[j >= 2 * edge] = 5  # bottom

    # Convert to image xyz
    a = 2.0 * i / edge
    b = 2.0 * j / edge
    one_arr = numpy.ones(len(a))
    for k in range(6):
        face_idx = face == k

        # Using the face_idx version of each is 50% quicker
        one_arr_idx = one_arr[face_idx]
        a_idx = a[face_idx]
        b_idx = b[face_idx]

        if k == 0:
            vals_to_use = [-one_arr_idx, 1.0 - a_idx, 3.0 - b_idx]
        elif k == 1:
            vals_to_use = [a_idx - 3.0, -one_arr_idx, 3.0 - b_idx]
        elif k == 2:
            vals_to_use = [one_arr_idx, a_idx - 5.0, 3.0 - b_idx]
        elif k == 3:
            vals_to_use = [7.0 - a_idx, one_arr_idx, 3.0 - b_idx]
        elif k == 4:
            vals_to_use = [b_idx - 1.0, a_idx - 5.0, one_arr_idx]
        elif k == 5:
            vals_to_use = [5.0 - b_idx, a_idx - 5.0, -one_arr_idx]

        xyz[face_idx] = numpy.array(vals_to_use).T

    # Convert to theta and pi
    x, y, z = xyz.T
    theta = numpy.arctan2(y, x)
    r = numpy.sqrt(x ** 2 + y ** 2)
    phi = numpy.arctan2(z, r)

    # Source img coords
    uf = (2.0 * edge * (theta + pi) / pi) % in_size[0]
    uf[uf == in_size[0]] = 0.0  # Wrap to pixel 0 (much faster than modulus)
    vf = (2.0 * edge * (pi / 2 - phi) / pi)

    # Mapping matrix
    out_pix[j, i, 0] = vf
    out_pix[j, i, 1] = uf

    print(out_pix[360, 800])

    map_x_32 = out_pix[:, :, 1]
    map_y_32 = out_pix[:, :, 0]
    return map_x_32, map_y_32


def equi2cubemap(imgIn):
    # imgIn = Image.open(sys.argv[1])
    # imgIn = cv2.imread(sys.argv[1])
    # inSize = imgIn.size
    inSize = imgIn.shape

    # map_x_32, map_y_32 = generate_mapping_data(inSize[0])
    map_x_32, map_y_32 = generate_mapping_data(inSize[1])
    # cubemap = cv2.remap(numpy.array(imgIn), map_x_32, map_y_32, cv2.INTER_LINEAR)
    cubemap = cv2.remap(imgIn, map_x_32, map_y_32, cv2.INTER_LINEAR)

    imgOut = Image.fromarray(cubemap)
    imgOut.save(sys.argv[1].split('.')[0] + "_out.png")
    imgOut.save("cubemap.png")
    imgOut.show()
    # outSize = cubemap.shape
    # print(cubemap)
    # top = Image.fromarray(cubemap[0:int(outSize[0]/3), int(outSize[1]/2-1):int(outSize[1]*3/4-1), :])
    # top.save("aaatop.png")
    # top.show()

    # top = cubemap[0:int(outSize[0]/3), int(outSize[1]/2-1):int(outSize[1]*3/4-1), :]
    # cv2.imwrite("aaatop.png", top)


def cubeToImg(coords, edge):
    if coords[0]=="Left":
        (x, y) = (int(edge*(coords[2]+1)/2), int(edge*(3-coords[3])/2) )
    elif coords[0]=="Front":
        (x, y) = (int(edge*(coords[1]+3)/2), int(edge*(3-coords[3])/2) )
    elif coords[0]=="Right":
        (x, y) = (int(edge*(5-coords[2])/2), int(edge*(3-coords[3])/2) )
    elif coords[0]=="Back":
        (x, y) = (int(edge*(7-coords[1])/2), int(edge*(3-coords[3])/2) )
    elif coords[0]=="Top":
        (x, y) = (int(edge*(3-coords[1])/2), int(edge*(1+coords[2])/2) )
    elif coords[0]=="Bottom":
        (x, y) = (int(edge*(3-coords[1])/2), int(edge*(5-coords[2])/2) )
    return (x, y)


def cot(angle):
    return 1/tan(angle)


def projection(theta,phi):
        if theta<0.615:
            return projectTop(theta,phi)
        elif theta>2.527:
            return projectBottom(theta,phi)
        elif phi <= pi/4 or phi > 7*pi/4:
            return projectLeft(theta,phi)
        elif phi > pi/4 and phi <= 3*pi/4:
            return projectFront(theta,phi)
        elif phi > 3*pi/4 and phi <= 5*pi/4:
            return projectRight(theta,phi)
        elif phi > 5*pi/4 and phi <= 7*pi/4:
            return projectBack(theta,phi)


def projectLeft(theta,phi):
        x = 1
        y = tan(phi)
        z = cot(theta) / cos(phi)
        if z < -1:
            return projectBottom(theta,phi)
        if z > 1:
            return projectTop(theta,phi)
        return ("Left",x,y,z)


def projectFront(theta,phi):
        x = tan(phi-pi/2)
        y = 1
        z = cot(theta) / cos(phi-pi/2)
        if z < -1:
            return projectBottom(theta,phi)
        if z > 1:
            return projectTop(theta,phi)
        return ("Front",x,y,z)


def projectRight(theta,phi):
        x = -1
        y = tan(phi)
        z = -cot(theta) / cos(phi)
        if z < -1:
            return projectBottom(theta,phi)
        if z > 1:
            return projectTop(theta,phi)
        return ("Right",x,-y,z)


def projectBack(theta,phi):
        x = tan(phi-3*pi/2)
        y = -1
        z = cot(theta) / cos(phi-3*pi/2)
        if z < -1:
            return projectBottom(theta,phi)
        if z > 1:
            return projectTop(theta,phi)
        return ("Back",-x,y,z)


def projectTop(theta,phi):
        # (a sin θ cos ø, a sin θ sin ø, a cos θ) = (x,y,1)
        a = 1 / cos(theta)
        x = tan(theta) * cos(phi)
        y = tan(theta) * sin(phi)
        z = 1
        return ("Top",x,y,z)


def projectBottom(theta,phi):
        # (a sin θ cos ø, a sin θ sin ø, a cos θ) = (x,y,-1)
        a = -1 / cos(theta)
        x = -tan(theta) * cos(phi)
        y = -tan(theta) * sin(phi)
        z = -1
        return ("Bottom",x,y,z)


def sphere2cubemap(theta, thi, face, edge):
    if face == 0:
        a = tan(theta) + 1.0
        b = 3.0 - tan(thi) * sqrt(1.0 + (1.0 - a)**2)
    elif face == 1:
        a = -1.0 / tan(theta) + 3.0
        b = 3.0 - tan(thi) * sqrt(1.0 + (a - 3.0)**2)
    elif face == 2:
        a = tan(theta) + 5.0
        b = 3.0 - tan(thi) * sqrt(1.0 + (a - 5.0)**2)
    elif face == 3:
        a = 7.0 - 1.0 / tan(theta)
        b = 3.0 - tan(thi) * sqrt((7.0 - a)**2 + 1.0)
    elif face == 4:
        b = 1.0 + sqrt(1 / (tan(thi)**2 * (tan(theta**2) + 1.0)))
        a = 5.0 + (b - 1.0) * tan(theta)
    elif face == 5:
        b = 5.0 - sqrt(1 / (tan(thi)**2 * (1.0 + tan(theta)**2)))
        a = 5.0 + (5.0 - b) * tan(theta)

    # j is row, i is col
    i = a * edge / 2.0
    j = b * edge / 2.0
    return [j, i]


def hhh(h, w, cubeImg):
    # [h, w] = img[:2]
    # h and w are for equi
    edge = int(w / 4)
    outImg = np.zeros([h, w, 3])
    for r in range(h):
        for c in range(w):
            [theta, thi] = equi2sphere(h, w, r, c)
            if thi < 0.615:
                face = 4
            elif thi > 2.527:
                face = 5
            elif theta <= pi/4 or theta > 7*pi/4:
                face = 0
            elif theta > pi/4 or theta <= 3*pi/4:
                face = 1
            elif theta > 3*pi/4 or theta <= 5*pi/4:
                face = 2
            elif theta > 5*pi/4 or theta <= 7*pi/4:
                face = 3

            if face < 4:
                z = cot(thi) / cos(theta)
                if z < -1:
                    face = 5
                elif z > 1:
                    face = 4

            [cubeRow, cubeCol] = sphere2cubemap(theta, thi, face, edge)
            biVal = bilinear_interpolation(cubeRow, cubeCol, edge*3, edge*4)

            for rgb in range(3):
                outImg[r][c][rgb] = cubeImg[0][2][rgb] * biVal[4] + \
                                    cubeImg[0][3][rgb] * biVal[5] + \
                                    cubeImg[1][2][rgb] * biVal[6] + \
                                    cubeImg[1][3][rgb] * biVal[7]
    cv2.imwrite('cubbbbbb.png', outImg)
