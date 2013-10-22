import cv2
import cv2.cv as cv
import numpy as np
import os, sys

os.chdir(os.path.dirname(sys.argv[0]))

if(len(sys.argv)==2):
	f=sys.argv[1]
else:
	print "Usage: colorHist.py imagename.png"
	sys.exit(-1)
 
img = cv2.imread(f)
h = np.zeros((300,512,3))
levels=256
bins = np.arange(levels).reshape(levels,1)
color = [ (255,0,0),(0,255,0),(0,0,255) ]
for ch, col in enumerate(color):
    hist_item = cv2.calcHist([img],[ch],None,[levels],[10,255])
    cv2.normalize(hist_item,hist_item,0,255,cv2.NORM_MINMAX)
    hist=np.int32(np.around(hist_item))
    pts = np.column_stack((bins*2,300-hist))
    cv2.polylines(h,[pts],False,col)
 
#h=np.flipud(h)
 
cv2.imshow(f,h)
cv2.waitKey(0)