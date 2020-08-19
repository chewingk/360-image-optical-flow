import numpy as np
import cv2

import evalution
from padimage import padimage
import flowio

# frame1 = "optical_flow_gt/0001_rgb.jpg"
# frame2 = "optical_flow_gt/0002_rgb.jpg"
# frame3 = "optical_flow_gt/0003_rgb.jpg"
#
# frame1Pad = padimage(frame1)
# frame3Pad = padimage(frame3)
# frame1ImgGray = cv2.cvtColor(frame1Pad, cv2.COLOR_RGB2GRAY)
# frame3ImgGray = cv2.cvtColor(frame3Pad, cv2.COLOR_RGB2GRAY)
#
# disFlow = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST)
# flow = disFlow.calc(frame3ImgGray, frame1ImgGray, None)
#
# interpolatedImage = evalution.frameInterpolation(flow, frame1ImgGray)
# out_img = interpolatedImage[:, 240:1200]
# gt = cv2.cvtColor(cv2.imread(frame2), cv2.COLOR_RGB2GRAY)
# fie = evalution.frameInterpolationError(out_img, gt)
# print(fie)
# cv2.imwrite('abc.jpg', out_img)
# # print(interpolatedImage.shape)

###################################################

frame1 = "optical_flow_gt/0001_rgb.jpg"
frame2 = "optical_flow_gt/0002_rgb.jpg"

# frame1Img = cv2.imread(frame1)
frame1Pad = padimage(frame1)
frame1ImgGray = cv2.cvtColor(frame1Pad, cv2.COLOR_RGB2GRAY)
# frame2Img = cv2.imread(frame2)
frame2Pad = padimage(frame2)
frame2ImgGray = cv2.cvtColor(frame2Pad, cv2.COLOR_RGB2GRAY)

disFlow = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
flow = disFlow.calc(frame1ImgGray, frame2ImgGray, None)
cut_flow = flow[:, 240:1200]
# print(cut_flow.shape)

gt = flowio.readFlowFile("optical_flow_gt/0001_opticalflow_forward.flo")
print(np.amax(gt))
exit()
# gt = flowio.readFlowFile("optical_flow_gt/0001_opticalflow_backward.flo")

# print(flow)
# print(flow.shape)

# epe = evalution.endPointError(gt, cut_flow)
# print(epe)
# ae = evalution.angularError(gt, cut_flow)
# print(ae)

errormap = np.zeros([gt.shape[0], gt.shape[1]])

diff = gt - cut_flow

for row in range(errormap.shape[0]):
    for col in range(errormap.shape[1]):
        # print(diff[row][col])
        errormap[row][col] = np.linalg.norm(diff[row][col])
        # print(errormap[row][col])
        # exit()
errormap = errormap / np.amax(errormap) * 255.0
cv2.imwrite('errormap.jpg', errormap.astype(int))
# print(np.amax(errormap))
