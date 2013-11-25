import cv2
import cv2.cv as cv
import os
import sys
import numpy as np
import atexit
import glob
from timeit import Timer
import math

'''
C:\Users\gfeng\Documents\GitHub\video2aoi>kernprof.py -l -v tempMatch2.py ITDS_s
ig\size1920calc.png itds_sig\sig5.png
No valid color plane specified; use all colors. -1
Location=(414, 219)
Score=0.131304845214
Wrote profile results to tempMatch2.py.lprof
Timer unit: 3.84954e-07 s

File: tempMatch2.py
Function: match at line 88
Total time: 1.80623 s

Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
    88                                           @profile
    89                                           def match():
    90
    91                                               #result = cv2.matchTemplate
(frame,tmp,cv.CV_TM_SQDIFF) # 0.2sec
    92         1      1318036 1318036.0     28.1      result = cv2.matchTemplate
(frame,sig,cv.CV_TM_SQDIFF_NORMED) # 0.2sec
    93                                               #t=Timer(myfunc, setup="imp
ort cv2; import cv2.cv as cv; import math; from __main__ import cosine_distance"
)
    94                                               #print t.timeit(1000)
    95
    96         1         4178   4178.0      0.1      minVal,maxVal,minLoc,maxLoc
 = cv2.minMaxLoc(result)
    97                                               ##
    98         1           11     11.0      0.0      windowName = "match"
    99                                               #cv2.rectangle(frame,(x,y),
(x+w,y+h),(0,255,0),2)
   100         1          202    202.0      0.0      cv2.rectangle(frame,(minLoc
[0],minLoc[1]),(minLoc[0]+h,minLoc[1]+w),(128,128,128),2)
   101
   102         1        10411  10411.0      0.2      print "Location="+str(minLo
c)
   103         1         3143   3143.0      0.1      print "Score="+str(minVal)
   104
   105         1        54914  54914.0      1.2      cv2.namedWindow(windowName)

   106         1        10004  10004.0      0.2      cv2.imshow(windowName, fram
e)
   107         1      3221507 3221507.0     68.7      cv2.waitKey()
   108
   109         1        69673  69673.0      1.5      cv2.destroyWindow(windowNam
e)

'''
#os.chdir(os.path.dirname(sys.argv[0]))

colorPlane = -1
try:
    colorPlane = int(sys.argv[3])
except:
    #print "No color plan specified; use all colors. " +str(colorPlane)
    pass

try:
    frameFileName = sys.argv[1]
    sigFileName = sys.argv[2]
except:
    print "tempmath.py frameFileName sigFileName [colorPlane 0-2]"
    #exit(0)
    sigFileName="itds_sig\\sig5.png"
    frameFileName="itds_sig\\size1920calc.png"



if 0<=colorPlane<=2:
    print "Using color plane " +str(colorPlane)
    frame= cv2.split(cv2.imread(frameFileName))[colorPlane]
    sig =  cv2.split(cv2.imread(sigFileName))[colorPlane]
else:
    print "No valid color plane specified; use all colors. " +str(colorPlane)
    frame= cv2.imread(frameFileName)
    sig =  cv2.imread(sigFileName)

if frame is None: 
    print "Error, can't find image file "+str(frameFileName)
    exit(0)
if sig is None:   
    print "Error, can't find template file "+str(sigFileName)
    exit(0)

    #frame= cv2.imread("ASU12_signatures/img.png")
#sig = cv2.imread("ASU12_signatures/signature_ASU12.png")
W=frame.shape[0]
H=frame.shape[1]
w=600
h=200

w=sig.shape[0]
h=sig.shape[1]

tmp=cv.CreateMat(w,h,cv.CV_32FC1)
x=66
y=577
tmp=frame[x:w, y:y+h]

width=W-w+1
height=H-h+1
#result = cv.CreateImage((width, height), 32,1)
result = np.zeros((width, height,3))

def cosine_distance(u, v):
    """
    Returns the cosine of the angle between vectors v and u. This is equal to
    u.v / |u||v|.
    """
    return np.dot(u, v) / (math.sqrt(np.dot(u, u)) * math.sqrt(np.dot(v, v)))

def myfunc():
    global frame, tmp, result
    x=np.random.randint(0, 1000, 1000)
    y=np.random.randint(0, 1000, 1000)
    v=frame[x,y]
    x=np.random.randint(0, 1000, 1000)
    y=np.random.randint(0, 1000, 1000)
    u=frame[x,y]
    cosine_distance(v,u)
    
def timeme():
    global frame, tmp, result
    cv2.absdiff(frame, frame) #0.0005

@profile
def match():

    #result = cv2.matchTemplate(frame,tmp,cv.CV_TM_SQDIFF) # 0.2sec 
    result = cv2.matchTemplate(frame,sig,cv.CV_TM_SQDIFF_NORMED) # 0.2sec 
    #t=Timer(myfunc, setup="import cv2; import cv2.cv as cv; import math; from __main__ import cosine_distance")
    #print t.timeit(1000)

    minVal,maxVal,minLoc,maxLoc = cv2.minMaxLoc(result)
    ##
    windowName = "match"
    #cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
    cv2.rectangle(frame,(minLoc[0],minLoc[1]),(minLoc[0]+h,minLoc[1]+w),(128,128,128),2)

    print "Location="+str(minLoc)
    print "Score="+str(minVal)

    cv2.namedWindow(windowName)
    cv2.imshow(windowName, frame)
    cv2.waitKey()

    cv2.destroyWindow(windowName)

if __name__ == "__main__":
    match()