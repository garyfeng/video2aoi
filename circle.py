import cv2
import cv2.cv as cv
import numpy as np
import sys

img = cv2.imread('img.png')
if img==None:
    print "cannot open ",filename

else:
    #img = cv2.medianBlur(img,3)
    #cimg = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
    #circles = cv2.HoughCircles(img[:,:,2],cv2.cv.CV_HOUGH_GRADIENT,1,10,param1=100,param2=30,minRadius=3,maxRadius=50)
    #circles = np.uint16(np.around(circles))
        
    # for i in circles[0,:]:
        # cv2.circle(img,(i[0],i[1]),i[2],(0,255,0),1) # draw the outer circle
        # cv2.circle(img,(i[0],i[1]),2,(0,0,255),3) # draw the center of the circle

        
    # Remember -> OpenCV stores things in BGR order
    #lowerBound = cv.Scalar(128, 128, 250);
    #upperBound = cv.Scalar(255, 255, 255);

    # this gives you the mask for those in the ranges you specified,
    # but you want the inverse, so we'll add bitwise_not...
    hsv_img = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    
    # CV's HSV values: In OpenCV, value range for  'hue', 'saturation' and 'value'  are respectively 0-179, 0-255 and 0-255.
    # Red crosses 0; we are looking at high value, pretty high saturation
    lowerBound = np.array([0, 130, 200],np.uint8);
    upperBound = np.array([5, 255, 255],np.uint8);
    cv_rgb_thresh1 = cv2.inRange(hsv_img, lowerBound, upperBound);
    
    lowerBound = np.array([175, 130, 200],np.uint8);
    upperBound = np.array([179, 255, 255],np.uint8);
    cv_rgb_thresh2 = cv2.inRange(hsv_img, lowerBound, upperBound);
    cv_rgb_thresh=np.bitwise_not(cv_rgb_thresh1+cv_rgb_thresh2)
    
    img = cv2.bitwise_and(img,img,mask = cv_rgb_thresh)

    #cv.Not(cv_rgb_thresh, cv_rgb_thresh);
    cv2.imshow('Red',cv_rgb_thresh)
    #cv2.waitKey(0)
    cv2.imshow('original',img)
    cv2.waitKey(0)
    #cv2.imwrite('output.png',cimg)
    cv2.destroyAllWindows()