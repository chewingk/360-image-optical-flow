# Optical Flow Computation for 360° Images
[Thesis Report](https://chewingk.github.io/docs/360FlowIcosahedron.pdf)
## Abstract
Virtual Reality is becoming more popular in recent years, so more researches have been made for 360° images. The computation of 360° optical flow has not been focused even though some researchers have used them for their researches. In this thesis, we present a 360° optical flow computation method, using Icosahedron projection for 360° images. We have also implemented 2 existing methods for comparison. We use a 360° image dataset with ground truth flows, to compare the performance for the 3 approaches. We use 3 common optical flow metrics: Endpoint Error, Angular Error and Frame Interpolation Error, and we also propose 2 new metrics for evaluating 360° flows: Great Circle Distance and Spherical Angular Error. We show that our approach can provide more accurate 360° flows, and the computation of 360° flows is worth digging deeper in the future.

## Instruction for Running Codes
### Import dependencies
```
import numpy as np
import cv2

from evalution import endPointError, flow_correction, angularError, sphericalEndPointError, sphericalAugularError,\
    frameInterpolation, frameInterpolationError
from padimage import padimage
from barymap import equiImg2EquiFlowBaryPreprocessed, equiImg2EquiFlowCubemapPreprocessed, equi2equi
```

### Create DISOpticalFlow instance
Mode can be ULTRAFAST, FAST or MEDIUM.
```
disflow = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
```
#### Read in two images from ```$IMG1_PATH``` and ```$IMG2_PATH```, then convert to grayscale value.
```
img1_gray = cv2.cvtColor(cv2.imread($IMG1_PATH), cv2.COLOR_RGB2GRAY)
img2_gray = cv2.cvtColor(cv2.imread($IMG2_PATH), cv2.COLOR_RGB2GRAY)
```
### Equirectanglar Projection
```
flow = equi2equi(img1_gray, img2_gray, disflow)
```
### Cubemap Projection
Run first line of code in a terminal (it pre-compute some arrays and saves as .npy files.), and the rest in Python
```
python cubemapPrePro.py
```
```
cubemapNumber = np.load('cubemapNumber.npy')
equiCubemapBiValSet = np.load('equiCubemapBiValSet.npy')
equiCubemapStartPointSet = np.load('equiCubemapStartPointSet.npy')
cubemapCartSet = np.load('cubemapCartSet.npy', allow_pickle=True)
cubemapFlowSet = np.load('cubemapFlowSet.npy')
flow = equiImg2EquiFlowCubemapPreprocessed(img1_gray, img2_gray, cubemapNumber, equiCubemapBiValSet,
                                           equiCubemapStartPointSet, cubemapCartSet, cubemapFlowSet, disflow)
```
### Icosahedron Projection
Run first line of code in a terminal (it pre-compute some arrays and saves as .npy files.), and the rest in Python
```
python icosahedronPrePro.py
```
```
triangleNumber = np.load('triangleNumber.npy')
equiBiValSet = np.load('equiBiValSet.npy')
equiStartPointSet = np.load('equiStartPointSet.npy')
triCartSet = np.load('triCartSet.npy', allow_pickle=True)
triangleFlowSet = np.load('flowSet.npy')
disflow = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
flow = equiImg2EquiFlowBaryPreprocessed(img1_gray, img2_gray, triangleNumber, equiBiValSet,
                                        equiStartPointSet, triCartSet, triangleFlowSet, disflow)
```
### Fix flow to avoid large flows
```
flow_correction(flow)
```
### Compare 2 flows ```$FLOW1``` and ```$FLOW2```
#### EPE
```
epe = endPointError($FLOW1, $FLOW2)
```
#### AE
```
ae = angularError($FLOW1, $FLOW2)
```
#### GCD
```
gcd = sphericalEndPointError($FLOW1, $FLOW2)
```
#### SAE
```
sae = sphericalAugularError($FLOW1, $FLOW2)
```
#### FIE
Flow from ```img1_gray``` to ```img2_gray```, use ```flow``` and ```img2_gray``` we can get an estimated\interpolated ```img1_gray```.
```
int_img1_gray = frameInterpolation(flow, img2_gray)
```

```
fie = frameInterpolationError(int_img1_gray, img1_gray)
```
