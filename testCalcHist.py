import cv2
import cv2.cv as cv
import numpy as np
import os, sys

os.chdir(os.path.dirname(sys.argv[0]))

if(len(sys.argv)==2):
    f=sys.argv[1]
else:
    print "Usage: testCalcHist.py imagename.png"
    sys.exit(-1)
    
img = cv2.imread(f)
h = np.zeros((300,256,3))
levels = 3
bins = np.arange(levels).reshape(levels,1)
color = [ (255,0,0),(0,255,0),(0,0,255) ]

for ch, col in enumerate(color):
    # Python: cv2.calcHist(images, channels, mask, histSize, ranges[, hist[, accumulate]]) -> hist
    hist_item = cv2.calcHist([img],[ch],None,[levels],[0,255])
    cv2.normalize(hist_item,hist_item,0,255,cv2.NORM_MINMAX)
    hist=np.int32(np.around(hist_item))
    pts = np.column_stack((bins * 256/levels,hist))
    cv2.polylines(h,[pts],False,col)

# gray
gimg=cv2.cvtColor(img, cv.CV_RGB2GRAY)
hist_item = cv2.calcHist([gimg],[0],None,[levels],[0,255])
hist_item2 = np.copy(hist_item)
cv2.normalize(hist_item,hist_item2,0,255,cv2.NORM_MINMAX)
hist=np.int32(np.around(hist_item2))
pts = np.column_stack((bins * 256/levels,hist))
cv2.polylines(h,[pts],False,(128,128,128))

h=np.flipud(h)

cv2.imshow('colorhist',h)
cv2.waitKey(0)