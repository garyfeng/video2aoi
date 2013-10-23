'''FrameEngine
Gary Feng, 2013
Princeton, NJ

This module implements various methods for detecting video frame changes 
and recognizing content on a video frame.
'''

import cv2
import cv2.cv as cv
import os
import os.path
import sys
import numpy as np
import glob
import logging
import re
import subprocess


# class for video comparisons
class FrameEngine:
    '''This engine uses OpenCV to implement a number of methods to detect/recognize video contents.

    To-dos:
        implement a faster algorithm
        -- sample_positions= a grid or a 2-d uniform random sample from a frame
            -- can be more strategic but a random or uniform scheme may work
            -- if there is a training sample (screenshots), select most informative positions
        -- signature = descripter(frame, sample_positions)
            -- descripter: simplest = the RGB or grayscale value at the point
            --    should also consider simple n-d descripters such as neighboring pixels, etc. 
            -- also create signatures of blank pages, etc. 
        -- given a new frame, calculate the cosine distance with each vector
            -- see http://stackoverflow.com/questions/2380394/simple-implementation-of-n-gram-tf-idf-and-cosine-similarity-in-python
            -- or use somethign like FLANN ... does it give distance?
            -- can also identify sub-screen changes
        -- fully automatic mode
            -- set an in/out threshold
            -- monitor changes in cosine similarity and detect abrupt changes
            -- save screenshots, match existing templates
            -- does additional processing if predefined templates are matched ... ocr, etc.
            -- (automatic layout analyses; text detection; image detection?)
            -- AOI generation
    '''
    lastFrame = None
    scrollImage = None
    frameChangeThreshold = 8.0
    blankThreshold = 10.0
    #matchTemplateThreshold =  5000000.0
    matchTemplateThreshold =  0.20

    def __init__ (self):
        self.clearScrollImage()
        return
    # ScrollImage is an image buffer to keep the reconstructed image when 
    # a region of a screen is defined as "scroll" in the YAML
    # we try to reconstruct the full image from the partial image in the current viewport 
    # over multiple scrolls. The goal is to define the AOI on the basis of the scrollImage.
    def clearScrollImage (self):
        self.scrollImage = None
    def setScrollImage (self, img):
        self.scrollImage = np.copy(img)  
    def getScrollImage (self):
        return self.scrollImage 
    def clearLastFrame (self):
        self.lastFrame = None
                            
    def isBlank (self, img):
        '''test to see if the frame is uniform; None if error'''
        if (len(img.shape)==3):
            try:
                img = cv2.cvtColor(img, cv.CV_RGB2GRAY)
            except:
                return None
        # now img is gray image        
        mean, std = cv2.meanStdDev(img)
        if (std<self.blankThreshold):
            return True
        else:
            return False
            
    def isDesktop (self, img):
        ''' compare grayscale histogram to that of a typical WinXP desktop. 
        It assumes the desktop is gray for on a grayscale image. 
        Obviously this is not a save assumption. 
        Maybe I should use the windows task bar instead.'''
        # we use a grayscale histogram with 3 levels :)
        if (len(img.shape)==3):
            try:
                img = cv2.cvtColor(img, cv.CV_RGB2GRAY)
            except:
                return None
        # now get the histogram
        levels=3
        hist_item = cv2.calcHist([img],[0],None,[levels],[0,255])
        cv2.normalize(hist_item,hist_item,0,255,cv2.NORM_MINMAX)
        hist=np.int32(np.around(hist_item))
        # desktop: a lot of gray
        if (hist[1][0]==255): return True
        else: return False
            
    def frameChanged (self, vframe, channel = 99):
        '''Compare the current frame to the last frame and return a boolean.
        The diff between the vframe and last vframe is compared to frameChangeThreshold.
        The optional channel parameters specifies which color channel is used to compare the images.
        '''
        # get image of the specified color channel: 
        if channel>=0 and channel<=2:
            vframe = cv2.split(vframe)[channel]
            
        if self.lastFrame is None:
            # lastFrame not set
            #self.lastFrame = vframe
            # the above is a reference; use numpy copy
            self.lastFrame = np.copy(vframe)
            logging.debug( "frameChanged: First frame or starting SkimmingMode")
            return True
        if self.lastFrame.shape != vframe.shape:
            # current frame is different from lastFrame in size or color depth; reset last frame
            #self.lastFrame = vframe
            # the above is a reference; use numpy copy
            self.lastFrame = np.copy(vframe)
            logging.error( "frameChanged: Error: lastFrame of different size")
            return True
        diffFrame = cv2.absdiff(vframe, self.lastFrame)
        #debug
        #print (str(self.frameChangeThreshold)+" - "+str( np.mean(diffFrame)))
        
        #self.lastFrame = vframe # update frame
        self.lastFrame = np.copy(vframe)
        if (np.mean(diffFrame) <self.frameChangeThreshold):
            logging.debug( "frameChanged: Change not big enough " +str( np.mean(diffFrame)))
            return False
        else:
            logging.debug( "frameChanged: Changed " + str( np.mean(diffFrame)))
            return True

    def featureMatch (self, template, r_threshold = 0.6):
        ''' use SURF/FLANN to map the template with an internal scrollImage, return the offset of template
        inside the scrollImage, plus the features; None if something goes wrong'''
        FLANN_INDEX_KDTREE = 1  # bug: flann enums are missing
        flann_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 4)

        if type(template) is not np.ndarray: 
            logging.error("featureMatch: error - input must be a valid image as Numpy array")
            return None                

        if self.scrollImage is None: self.scrollImage = np.copy(template)
        # if scrollImage is blank
        if self.isBlank(self.scrollImage): self.scrollImage = np.copy(template)

#        # potential code for integrating more openCV options here; 
#        # code from find_obj.py example
#        # start feature extraction
#        if chunks[0] == 'sift':
#            detector = cv2.SIFT()
#            norm = cv2.NORM_L2
#        elif chunks[0] == 'surf':
#            detector = cv2.SURF(800)
#            norm = cv2.NORM_L2
#        elif chunks[0] == 'orb':
#            detector = cv2.ORB(400)
#            norm = cv2.NORM_HAMMING
        
        detector = cv2.SURF(1000)
        kp1, desc1 = detector.detectAndCompute(template, None)
        kp2, desc2 = detector.detectAndCompute(self.scrollImage, None)

        if desc1 is None: 
            logging.error("featureMatch: error no feature can be found for template")
            return None
        if desc2 is None: 
            logging.error("featureMatch: error no feature can be found for self.scrollImage")
            # if the scrollImage is blank, we need to get somethings to start with. copy the template
            if desc1 is not None: self.scrollImage = np.copy(template)
            else: self.scrollImage=None
            return None
       
        desc1.shape = (-1, detector.descriptorSize())
        desc2.shape = (-1, detector.descriptorSize())
        logging.debug('featureMatch: image - '+str(len(kp1))+' features, template - '+str(len(kp2))+' features')
        
        # FLANN
#        # potential code for integrating more openCV options here
#        if norm == cv2.NORM_L2:
#            flann_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
#        else:
#            flann_params= dict(algorithm = FLANN_INDEX_LSH,
#                               table_number = 6, # 12
#                               key_size = 12,     # 20
#                               multi_probe_level = 1) #2
#        matcher = cv2.FlannBasedMatcher(flann_params, {})  # bug : need to pass empty dict (#1329)
        
        flann = cv2.flann_Index(desc2, flann_params)
        idx2, dist = flann.knnSearch(desc1, 2, params = {}) # bug: need to provide empty dict
        #print idx2
        #print str(dist[:,1])
        if any([x<=0 for x in dist[:,1]]): 
            logging.error("featureMatch: error - zeros in dist[:,1]")
            return None
        mask = dist[:,0] / dist[:,1] < r_threshold
        idx1 = np.arange(len(desc1))
        pairs = np.int32( zip(idx1, idx2[:,0]) )
        
        m=pairs[mask]
        #print m
        if (len(m)<5): 
            logging.error("featureMatch: error # of matching features <5")
            return None
        matched_p1 = np.array([kp1[i].pt for i, j in m])
        matched_p2 = np.array([kp2[j].pt for i, j in m])
        
        H, status = cv2.findHomography(matched_p1, matched_p2, cv2.RANSAC, 5.0)
        logging.debug( 'featureMatch: %d / %d  inliers/matched' % (np.sum(status), len(status)))

        if H is None: 
            logging.error("featureMatch: error no match was found; reset self.scrollImage")
            return None

        # construct the scrollImage
        h1, w1 = template.shape[:2]
        h2, w2 = self.scrollImage.shape[:2]
        corners = np.float32([[0, 0], [w1, 0], [w1, h1], [0, h1]])
        corners = np.int32( cv2.perspectiveTransform(corners.reshape(1, -1, 2), H).reshape(-1, 2) )
        # find the space needed to draw the overlay picture
        x1=min([x for x,y in corners])
        x1=min(x1,0)
        x2=max([x for x,y in corners])
        x2=max(x2, w2, w1)
        visw = x2-x1
        y1=min([y for x,y in corners])
        y1=min(y1,0)
        y2=max([y for x,y in corners])
        y2=max(y2, h2, h1)
        vish = y2-y1
        
        if(vish>3000 or visw>3000 or vish<=0 or visw <=0):
            # very unlikely size, sign of mismatch
            logging.error("featureMatch: error shape of vis="+str((vish, visw))+ " returning None")
            return None
        vis = np.zeros((vish+1, visw+1), np.uint8)                
        px=min([x for x,y in corners])
        py=min([y for x,y in corners])
        # print x1, y1, x2, y2, visw, vish, px,py
        # @@@ notice the order: the new image will overwrite the existing image
        try:
            vis[0-y1:h2-y1, 0-x1:w2-x1] = self.scrollImage
        except:
            logging.error("featureMatch: error, scrollImage size="+str(self.scrollImage.shape)+" vis size="+str(vis.shape)+ "origin="+str((-y1,-y2)))
        try:
            vis[py-y1:py-y1+h1, px-x1:px-x1+w1] = template
        except:
            logging.error("featureMatch: error, template size="+str(template.shape)+" vis size="+str(vis.shape)+ " origin="+str((px,py))+" "+str((py-y1,py-y1+h1, px-x1,px-x1+w1)))
        
        #vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
        self.scrollImage = np.copy(vis)       
        #all goes well, return the (x,y) of the offset
        #@@@ assuming no other transformations than shifting, which should be ok with screenshots
        # offset, match, txtScrollImage
        return ((int(round(H[0,2])), int(round(H[1,2]))), (np.sum(status), len(status)))

            
    def findMatch (self, image, template, threshold=-99):
        '''find template in image; return (location, minV) if found, or None is not found'''
        minV,maxV,minL ,maxL = None, None, None, None
        if threshold == -99: threshold = self.matchTemplateThreshold    # default threshold if not specified
        if not len(image.shape)==len(template.shape):
            # different color depth
            logging.error("findMatch: image and template are of different color depths")
            return None
        try:
            #res = cv2.matchTemplate(image,template,cv.CV_TM_SQDIFF)
            res = cv2.matchTemplate(image,template,cv.CV_TM_SQDIFF_NORMED)
            minV,maxV,minL ,maxL = cv2.minMaxLoc(res)
        except:
            # most likely the images have different format
            logging.error("findMatch: matchTemplate failed")
            return None
        if minL is None:
            #logging.debug("findMatch: matchTemplate returns nothing")
            return None

        if (minV<threshold):
            # ignore small differences
            logging.debug("findMatch: found match at "+str(minL)+", minVal="+str(minV)+"<threshold="+str(threshold))
            return minL, minV
        else:
            logging.debug("findMatch: no match found; minVal="+str(minV)+">threshold="+str(threshold))
            return None
