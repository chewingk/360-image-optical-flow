'''
python by:  youngjung uh, Clova ML, Naver
contact:    youngjung.uh@navercorp.com
date:       17 Dec 2018
-------------------------------------------------------------------
---- below comment came from the original (readFlowFile.m) -------
---- below comment came from the original (writeFlowFile.m) -------
-------------------------------------------------------------------
readFlowFile read a flow file FILENAME into 2-band image IMG
writeFlowFile writes a 2-band image IMG into flow file FILENAME
  According to the c++ source code of Daniel Scharstein
  Contact: schar@middlebury.edu
  Author: Deqing Sun, Department of Computer Science, Brown University
  Contact: dqsun@cs.brown.edu
  $Date: 2007-10-31 15:36:40 (Wed, 31 Oct 2006) $
Copyright 2007, Deqing Sun.
                        All Rights Reserved
Permission to use, copy, modify, and distribute this software and its
documentation for any purpose other than its incorporation into a
commercial product is hereby granted without fee, provided that the
above copyright notice appear in all copies and that both that
copyright notice and this permission notice appear in supporting
documentation, and that the name of the author and Brown University not be used in
advertising or publicity pertaining to distribution of the software
without specific, written prior permission.
THE AUTHOR AND BROWN UNIVERSITY DISCLAIM ALL WARRANTIES WITH REGARD TO THIS SOFTWARE,
INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR ANY
PARTICULAR PURPOSE.  IN NO EVENT SHALL THE AUTHOR OR BROWN UNIVERSITY BE LIABLE FOR
ANY SPECIAL, INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
'''

import os
import numpy as np
from struct import pack, unpack


def readFlowFile(fname):
    '''
    args
        fname (str)
    return
        flow (numpy array) numpy array of shape (height, width, 2)
    '''

    TAG_FLOAT = 202021.25  # check for this when READING the file

    ext = os.path.splitext(fname)[1]

    assert len(ext) > 0, ('readFlowFile: extension required in fname %s' % fname)
    assert ext == '.flo', exit('readFlowFile: fname %s should have extension ''.flo''' % fname)

    try:
        fid = open(fname, 'rb')
    except IOError:
        print('readFlowFile: could not open %s', fname)

    tag     = unpack('f', fid.read(4))[0]
    width   = unpack('i', fid.read(4))[0]
    height  = unpack('i', fid.read(4))[0]

    assert tag == TAG_FLOAT, ('readFlowFile(%s): wrong tag (possibly due to big-endian machine?)' % fname)
    assert 0 < width and width < 100000, ('readFlowFile(%s): illegal width %d' % (fname, width))
    assert 0 < height and height < 100000, ('readFlowFile(%s): illegal height %d' % (fname, height))

    nBands = 2

    # arrange into matrix form
    flow = np.fromfile(fid, np.float32)
    flow = flow.reshape(height, width, nBands)

    fid.close()

    return flow


def writeFlowFile(img, fname):
    TAG_STRING = 'PIEH'    # use this when WRITING the file

    ext = os.path.splitext(fname)[1]

    assert len(ext) > 0, ('writeFlowFile: extension required in fname %s' % fname)
    assert ext == '.flo', exit('writeFlowFile: fname %s should have extension ''.flo''', fname)

    height, width, nBands = img.shape

    assert nBands == 2, 'writeFlowFile: image must have two bands'

    try:
        fid = open(fname, 'wb')
    except IOError:
        print('writeFlowFile: could not open %s', fname)

    # write the header
    # fid.write(TAG_STRING.encode(encoding='utf-8', errors='strict'))
    # code = unpack('f', bytes(TAG_STRING, 'utf-8'))[0]
    # fid.write(pack('f', code))
    fid.write(bytes(TAG_STRING, 'utf-8'))
    fid.write(pack('i', width))
    fid.write(pack('i', height))

    # arrange into matrix form
    tmp = np.zeros((height, width*nBands), np.float32)

    tmp[:, np.arange(width) * nBands] = img[:, :, 0]
    tmp[:, np.arange(width) * nBands + 1] = np.squeeze(img[:, :, 1])

    fid.write(bytes(tmp))

    fid.close()
