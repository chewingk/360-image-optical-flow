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

# frame1 = "optical_flow_gt/0001_rgb.jpg"
# frame2 = "optical_flow_gt/0002_rgb.jpg"
#
# # frame1Img = cv2.imread(frame1)
# frame1Pad = padimage(frame1)
# frame1ImgGray = cv2.cvtColor(frame1Pad, cv2.COLOR_RGB2GRAY)
# # frame2Img = cv2.imread(frame2)
# frame2Pad = padimage(frame2)
# frame2ImgGray = cv2.cvtColor(frame2Pad, cv2.COLOR_RGB2GRAY)
#
# disFlow = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
# flow = disFlow.calc(frame1ImgGray, frame2ImgGray, None)
# cut_flow = flow[:, 240:1200]
# # print(cut_flow.shape)
#
# gt = flowio.readFlowFile("optical_flow_gt/0001_opticalflow_forward.flo")
# print(np.amax(gt))
# exit()
# # gt = flowio.readFlowFile("optical_flow_gt/0001_opticalflow_backward.flo")
#
# # print(flow)
# # print(flow.shape)
#
# # epe = evalution.endPointError(gt, cut_flow)
# # print(epe)
# # ae = evalution.angularError(gt, cut_flow)
# # print(ae)
#
# errormap = np.zeros([gt.shape[0], gt.shape[1]])
#
# diff = gt - cut_flow
#
# for row in range(errormap.shape[0]):
#     for col in range(errormap.shape[1]):
#         # print(diff[row][col])
#         errormap[row][col] = np.linalg.norm(diff[row][col])
#         # print(errormap[row][col])
#         # exit()
# errormap = errormap / np.amax(errormap) * 255.0
# cv2.imwrite('errormap.jpg', errormap.astype(int))
# print(np.amax(errormap))
from flowio import readFlowFile
import flow_vis as fv
from evalution import endPointError, flow_correction
import matplotlib.pyplot as plt
img = cv2.imread('GTFlow/hotel/0001_rgb.jpg')
img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
img2 = cv2.imread('GTFlow/hotel/0001_rgb.jpg')
flow = readFlowFile('GTFlow/room1/0001_opticalflow_forward.flo')
# flow2 = readFlowFile('equi/room1/0001.flo')
# flow2 = readFlowFile('cubemap/room1/0001.flo')
flow2 = readFlowFile('icosahedron/room1/0001.flo')
flow2corr = flow_correction(flow2)
# h, w = flow.shape[:2]
# # print(endPointError(flow_correction(flow), flow_correction(flow2)))
# for row in range(h):
#     for col in range(w):
#         # u = flow[row, col, 0]
#         # v = flow[row, col, 1]
#         u = flow2[row, col, 0]
#         v = flow2[row, col, 1]
#         if np.sqrt(u**2 + v**2) > 8:
#         # if np.linalg.norm(flow2[row, col]):
#             # flow[row, col] = [0, 0]
#             flow2[row, col] = [0, 0]

# cv2.imwrite('testout/bbbbbb.jpg', fv.flow_to_color(flow2, clip_flow=12))
# cv2.imwrite('testout/aaaaaa.jpg', fv.flow_to_color(flow, clip_flow=12))
#
# hsv = np.zeros(img.shape, dtype=np.uint8)
# hsv[..., 1] = 255
#
# mag, ang = cv2.cartToPolar(flow2[..., 0], flow2[..., 1])
# hsv[..., 0] = ang * 180 / np.pi / 2
# hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
# bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
# # cv2.imshow("colored flow", bgr)
# # cv2.waitKey(0)
# # cv2.destroyAllWindows()
# cv2.imwrite('testout/aaaaaa.jpg', bgr)
# step = 3
# plt.quiver(np.arange(0, flow2.shape[1], step), np.arange(flow2.shape[0], 0, -step),
#            flow[::step, ::step, 0], flow[::step, ::step, 1])
# plt.savefig('testout/cccccc.jpg')
cv2.imwrite('testout/bbbbbb.jpg', fv.draw_flow(img_gray, flow2corr))
