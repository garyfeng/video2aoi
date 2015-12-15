import cv2
import cv2.cv as cv
import os
import sys
import numpy as np
import atexit

atexit.register(cv2.destroyAllWindows)

os.chdir(os.path.dirname(sys.argv[0]))
f = "GRE_07D8.avi"
video = cv2.VideoCapture(f)
windowName = "My video"
cv2.namedWindow(windowName)
cv2.resizeWindow(windowName, 800, 600)

lastFrame = None
diffFrame = None

nFrames = int(video.get(cv.CV_CAP_PROP_FRAME_COUNT))
fps = video.get(cv.CV_CAP_PROP_FPS)
waitPerFrameInMillisec = int(1 / fps * 1000 / 1)

counter = 0
while video.grab():
        counter += 1
        flag, frame = video.retrieve()
        if flag:
                # gray_frm = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                # cv2.imwrite('frame_'+str(counter)+'.png',gray_frm)
                # compare frames
                if lastFrame is None:
                        lastFrame = np.copy(frame)
                        w = int(video.get(cv.CV_CAP_PROP_FRAME_WIDTH))
                        h = int(video.get(cv.CV_CAP_PROP_FRAME_HEIGHT))
                        diffFrame = cv.CreateMat(w, h, cv.CV_32FC1)
                        diffFrame = cv.fromarray(frame)
                # this seems to mess up the color space
                # diffFrame = np.abs(frame-lastFrame)
                # cv2.imshow(windowName, diffFrame)
                cv.AbsDiff(cv.fromarray(
                           frame), cv.fromarray(lastFrame), diffFrame)
                cv2.imshow(windowName, np.asarray(diffFrame))
                cv2.waitKey(waitPerFrameInMillisec - 2)

        lastFrame = np.copy(frame)

cv2.waitKey()
cv2.destroyWindow(windowName)
