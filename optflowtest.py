import cv2
import numpy as np
import sys
import flow_vis as fv

# feature_params = dict(maxCorners=100,
#                       qualityLevel=0.3,
#                       minDistance=7,
#                       blockSize=7)
# lk_params = dict(winSize=(15, 15),
#                  maxLevel=2,
#                  criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
# color = np.random.randint(0, 255, (100, 3))

img1 = cv2.imread(sys.argv[1])
img1_gray = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
img2 = cv2.imread(sys.argv[2])
img2_gray = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)

mask = np.zeros_like(img1)
# frame = img2
#
# p0 = cv2.goodFeaturesToTrack(img1_gray, mask=None, **feature_params)
#
# p1, st, err = cv2.calcOpticalFlowPyrLK(img1_gray, img2_gray, p0, None, **lk_params)
#
# good_new = p1[st == 1]
# good_old = p0[st == 1]
#
# for i, (new, old) in enumerate(zip(good_new, good_old)):
#     a, b = new.ravel()
#     c, d = old.ravel()
#     mask = cv2.line(mask, (a, b), (c, d), color[i].tolist(), 2)
#     frame = cv2.circle(frame, (a, b), 5, color[i].tolist(), -1)
#
# outimg = cv2.add(frame, mask)
#
# cv2.imwrite('ttt.jpg', outimg)

# flow = cv2.calcOpticalFlowFarneback(img1_gray, img2_gray, None, 0.5, 5, 11, 5, 5, 1.1, 0)
#####
aaa = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST)
flow = aaa.calc(img1_gray, img2_gray, None)
print(flow.shape)

mask_rgb = fv.flow_to_color(flow)

#####
# mag, ang = cv2.cartToPolar(flow[:, :, 0], flow[:, :, 1])
# mask[:, :, 0] = (ang * 180) / (np.pi * 2)
# mask[:, :, 1] = 255
# mask[:, :, 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
# mask_rgb = cv2.cvtColor(mask, cv2.COLOR_HSV2BGR)

# dense_flow = cv2.addWeighted(img1, 1, mask_rgb, 2, 0)

# cv2.imwrite('ttt.jpg', dense_flow)

cv2.imwrite('ooo2.jpg', mask_rgb)

