import cv2

vid = cv2.VideoCapture('test360.mp4')

success, img = vid.read()
count = 0

while success:
    cv2.imwrite('360images//frame{}.jpg'.format(count), img)
    success, img = vid.read()
    print('Next')
    count += 1
