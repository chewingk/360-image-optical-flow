import numpy as np
import sys


def readFlo(filename):
    with open(filename, 'rb') as f:
        magic, = np.fromfile(f, np.float32, count=1)
        if 202021.25 != magic:
            print('Magic number incorrect. Invalid .flo file')
        else:
            w, h = np.fromfile(f, np.int32, count=2)
            print(f'Reading {h} x {w} flo file')
            data = np.fromfile(f, np.float32, count=2*w*h)
            # Reshape data into 3D array (columns, rows, bands)
            data2D = np.resize(data, (h, w, 2))
            # print(data2D)
            # print(data2D.shape)
            return data2D


if __name__ == '__main__':
    readFlo(sys.argv[1])
