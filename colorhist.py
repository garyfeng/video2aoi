import cv2
import cv2.cv as cv
import numpy as np
import os, sys

class colorHist:

    """A class to plot color histogram"""
    
    levels = 256
    img = None
    
    def __init__ (self):
        #self.clearScrollImage()
        return
    
    def setLevels (levels):
        """set the global levels para"""
        self.levels = int(levels)
        return
        
    def colorHistVec (img, levels = 256):
        #img = cv2.imread(f)
        self.img = img
        bins = np.arange(levels).reshape(levels,1)
        color = [ (255,0,0),(0,255,0),(0,0,255) ]
        vec = None
        
        for ch, col in enumerate(color):
            # Python: cv2.calcHist(images, channels, mask, histSize, ranges[, hist[, accumulate]]) -> hist
            hist_item = cv2.calcHist([img],[ch],None,[levels],[0,255])
            cv2.normalize(hist_item,hist_item,0,255,cv2.NORM_MINMAX)
            hist=np.int32(np.around(hist_item))
            # concatinate the vector
            if (vec is None):
                vec = hist
            else:
                vec = np.hstack((vec, hist))
        return vec
        
    def plotColorHist (img, levels = 256):
        hist = self.colorHistVec 
        h = np.zeros((300,256,3))
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


if __name__ == "__main__":

    ##########################
    # global/main:
    ###########################################################
    # Starting

    os.chdir(os.path.dirname(sys.argv[0]))

    if(len(sys.argv)==2):
        f=sys.argv[1]
    else:
        print "Usage: testCalcHist.py imagename.png"
        sys.exit(-1)
    
