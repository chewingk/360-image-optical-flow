import numpy as np
import cv2

from flowio import readFlowFile
from evalution import endPointError, flow_correction, angularError
from padimage import padimage

# gt = readFlowFile('GTFlow/hotel/0001_opticalflow_forward.flo')
# gtt = readFlowFile('GTFlow/hotel/0002_opticalflow_forward.flo')
# dot = gt * gtt
# dotsum = np.sum(dot, axis=2)
#
# gtabs = np.linalg.norm(gt, axis=2)
# gttabs = np.linalg.norm(gtt, axis=2)
#
# print([gt[0, 0], gtt[0, 0]])
# print(dot[0, 0])
# print(dotsum[0, 0])
# print(gtabs[0, 0])
#
# fin = np.sum(dot, axis=2) / (np.linalg.norm(gt, axis=2) * np.linalg.norm(gtt, axis=2))
# # print(fin[0])
#
# print(np.rad2deg(np.nanmean(np.arccos(fin))))

epesum = []
aesum = []
# aaa = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST)
# aaa = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)
aaa = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
# ignore first image and last image
jpg = '_rgb.jpg'
direction = 'forward'
maxNum = 179

# # apartment
# folder = 'GTFlow/apartment'

# # office4
# folder = 'GTFlow/office4'

# # room1
# folder = 'GTFlow/room1'

# # hotel
# folder = 'GTFlow/hotel'

# # office0
# folder = 'GTFlow/office0'

# room0
folder = 'GTFlow/room0'

for idx in range(maxNum):
    if idx < 10:
        filename = f'{folder}/000{idx}_opticalflow_{direction}.flo'
        img1 = f'{folder}/000{idx}{jpg}'
        img2 = f'{folder}/00{idx+1}{jpg}' if idx == 9 else f'{folder}/000{idx+1}{jpg}'
    elif idx < 100:
        filename = f'{folder}/00{idx}_opticalflow_{direction}.flo'
        img1 = f'{folder}/00{idx}{jpg}'
        img2 = f'{folder}/0{idx+1}{jpg}' if idx == 99 else f'{folder}/00{idx+1}{jpg}'
    else:
        filename = f'{folder}/0{idx}_opticalflow_{direction}.flo'
        img1 = f'{folder}/0{idx}{jpg}'
        img2 = f'{folder}/0{idx+1}{jpg}'
    # print(img2)
    img1_gray = cv2.cvtColor(cv2.imread(img1), cv2.COLOR_RGB2GRAY)
    img2_gray = cv2.cvtColor(cv2.imread(img2), cv2.COLOR_RGB2GRAY)
    calcFlow = flow_correction(aaa.calc(padimage(img1_gray), padimage(img2_gray), None))
    correctedGT = flow_correction(readFlowFile(filename))
    epe = endPointError(correctedGT, calcFlow[:, 240:1200])
    ae = angularError(correctedGT, calcFlow[:, 240:1200])
    # print(epe)
    epesum.append(epe)
    aesum.append(ae)

print(folder)
print(np.average(np.array(epesum)))
print(np.average(np.array(aesum)))
