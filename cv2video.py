import cv2
import cv2.cv as cv
import os
import os.path
import sys
import numpy as np
#import atexit
import glob
import logging
import re
import subprocess

import tesseract
from HTMLParser import HTMLParser
from htmlentitydefs import name2codepoint

# from csv import reader

import yaml

# create a subclass and override the handler methods
class myTessHTMLParser(HTMLParser):
    box=None
    word=""
    image=None
    
    def __init__(self, img):
        # need to test to see if this is a numpy image that is fed to Tesseract (before turning into IPLimage)
        # if not we need to convert it to an numpy/cv2 image
        self.image = img
            
    def handle_starttag(self, tag, attrs):
        # print "Start tag:", tag
        #self.box = None
        for attr in attrs:
            # print "     attr:", attr
            # if (attr[0] in s): print attr[0]+": "+ attr[1]
            if (attr[0]=="title"):
                b=attr[1].split(" ")
                if (b[0]!="bbox"): 
                    continue
                try:
                    self.box = [int(xx) for xx in b[1:]]
                except:
                    self.box=None
        # now draw box
        if self.box is not None:
            try:
                cv2.rectangle(self.image,(self.box[0], self.box[1]),(self.box[2], self.box[3]),(0,128,0),1)
            except:
                pass
        else:
            #print "box is not valid"
            pass

    def handle_endtag(self, tag):
        #print "End tag  :", tag
        pass
    def handle_data(self, data):
        # print data
        if self.box is not None:
            try:
                cv2.putText(self.image, data, (self.box[0], self.box[1]), cv2.FONT_HERSHEY_PLAIN, 1.0, 0, thickness=1)
            except:
                pass
        pass
    def handle_comment(self, data):
        #print "Comment  :", data
        pass
    def handle_entityref(self, name):
        c = unichr(name2codepoint[name])
        #print "Named ent:", c
        if self.box is not None:
            try:
                cv2.putText(self.image, c, (self.box[0], self.box[1]), cv2.FONT_HERSHEY_PLAIN, 1.0, 0, thickness=1)
            except:
                pass
        pass
        
    def handle_charref(self, name):
        if name.startswith('x'):
            c = unichr(int(name[1:], 16))
        else:
            c = unichr(int(name))
        #print "Num ent  :", c
        if self.box is not None:
            try:
                cv2.putText(self.image, c, (self.box[0], self.box[1]), cv2.FONT_HERSHEY_PLAIN, 1.0, 0, thickness=1)
            except:
                pass
        pass
            
    def handle_decl(self, data):
        #print "Decl     :", data
        pass

    def getImage(self):
        return self.image

# create a subclass and override the handler methods
class TessHTMLParser(HTMLParser):
    box=None; ly0=0; ly1=0
    word=""; id=""; lid=""
    # to output AOI coordinates in the original space
    ocrZoomRatio = 3; ocrOffsetX=0; ocrOffsetY=0; ocrPageTitle = "OCR Page"
    callbackfunc = None # for constructing the aoilist
    
    def handle_starttag(self, tag, attrs):
        #print "Start tag:", tag
        #self.box = None
        for attr in attrs:
            # print "     attr:", attr
            # if (attr[0] in s): print attr[0]+": "+ attr[1]
            # update the ID  
            if (attr[0]=="id"):
                self.id=str(attr[1])
            if (attr[0]=="title"):
                # having to deal with an anormaly of Tesseract:
                # <div class='ocr_page' id='page_1' title='image "; bbox 0 0 3030 1665; ppageno 0'>
                dump=attr[1].split("; ")
                for d in dump:
                    b=d.split(" ")
                    if (b[0]!="bbox"): 
                        continue
                    try:
                        # adjust for the scale and shift of the ocr screen
                        self.box = [int(xx)/self.ocrZoomRatio for xx in b[1:]]
                        self.box[0]+=self.ocrOffsetX; self.box[2]+=self.ocrOffsetX
                        self.box[1]+=self.ocrOffsetY; self.box[3]+=self.ocrOffsetY
                    except:
                        self.box=None
            # if this is a line box, update the line parameters
            if self.box is not None and self.id.startswith("line"):
                self.ly0 = self.box[1]
                self.ly1 = self.box[3]
                self.lid = self.id
                
        # do not log if it's a word
        if (self.box is not None) and (self.id is not None) and (not self.id.startswith("word")):
            logging.info( "AOI: pageTitle="+self.ocrPageTitle+"\tid='"+str(self.id)+"'\tbox="+str(self.box)+"\tcontent='"+str(self.id)+"'")
            if self.callbackfunc is not None:
                self.callbackfunc((self.ocrPageTitle, self.id, self.id, self.box[0], self.box[1], self.box[2], self.box[3]))
        return

    def handle_endtag(self, tag):
        #print "End tag  :", tag
        return
    def handle_data(self, data):
        #print data
        if self.box is not None:
            # log if this is a word and is not empty space
            if self.id.startswith("word") and re.search(r'\S', data) is not None :
                logging.info( "AOI: pageTitle="+self.ocrPageTitle+"\tid='"+str(self.id)+"'\tlineid="+str(self.lid)+"\tbox="+str(self.box)+"\tlinebox=["+str(self.box[0])+", "+str(self.ly0)+", "+str(self.box[2])+", "+str(self.ly1)+"]\tcontent='"+str(data)+"'")
                if self.callbackfunc is not None:
                    self.callbackfunc((self.ocrPageTitle, self.id, str(data), self.box[0], self.ly0, self.box[2], self.ly1))
        return
    def handle_comment(self, data):
        #print "Comment  :", data
        return
    def handle_entityref(self, name):
        c = unichr(name2codepoint[name])
        #print "Named ent:", c
        if self.box is not None:
            # log if this is a word
            if self.id.startswith("word"):
                logging.info( "AOI: pageTitle="+self.ocrPageTitle+"\tid='"+str(self.id)+"'\tbox="+str(self.box)+"\tcontent='"+str(c)+"'")
                if self.callbackfunc is not None:
                    self.callbackfunc((self.ocrPageTitle, self.id, str(c), self.box[0], self.box[1], self.box[2], self.box[3]))
            else:
                logging.info( "AOI error: handle_entityref expecting a word element here")
        return
        
    def handle_charref(self, name):
        if name.startswith('x'):
            c = unichr(int(name[1:], 16))
        else:
            c = unichr(int(name))
        #print "Num ent  :", c
        if self.box is not None:
            # log if this is a word
            if self.id.startswith("word"):
                logging.info( "AOI: pageTitle="+self.ocrPageTitle+"\tid='"+str(self.id)+"'\tbox="+str(self.box)+"\tcontent='"+str(c)+"'")
                if self.callbackfunc is not None:
                    self.callbackfunc((self.ocrPageTitle, self.id, str(c),  self.box[0], self.box[1], self.box[2], self.box[3]))
            else:
                logging.info( "AOI error: handle_charref expecting a word element here")
        return
            
    def handle_decl(self, data):
        #print "Decl     :", data
        return


    def getImage(self):
        # return self.image
        return 
        
# Tesseract OCR routines
class TessEngine:
    tess = None
    ocrZoomRatio = 3
    boxen = None; html=None; text=None; confidence=None
    image=None; parser=None
    
    def __init__ (self):
            
        self.tess = tesseract.TessBaseAPI()
        self.tess.Init(".","eng",tesseract.OEM_DEFAULT)

    def image2txt (self, img):
        '''Takes an img (in cv2/numpy format) and OCR using Tesseract'''
        self.boxen=None
        self.html =None
        self.confidence = None
        # convert to gray, and zoom x 3
        if img is None: return ""
        if (len(img.shape)==3):
            img = cv2.cvtColor(img, cv.CV_RGB2GRAY)
        self.image = cv2.resize(img,(0,0), fx= self.ocrZoomRatio, fy=self.ocrZoomRatio )
        # Tesseract wants a grayscale IPLimage file. Can't use RGB or numpy array.
        bitmap = cv.CreateImageHeader((self.image.shape[1], self.image.shape[0]), cv.IPL_DEPTH_8U, 1)
        cv.SetData(bitmap, self.image.tostring(), self.image.dtype.itemsize * 1 * self.image.shape[1])

        #api.SetPageSegMode(tesseract.PSM_SINGLE_WORD)
        self.tess.SetPageSegMode(tesseract.PSM_AUTO)
        tesseract.SetCvImage(bitmap,self.tess)
        
        self.text=self.tess.GetUTF8Text()
        self.confidence=self.tess.MeanTextConf()
        return self.text


    def getText (self):
        return self.text
    def getConfidence(self):
        return self.confidence
    def getBoxen(self):
        self.boxen = self.tess.GetBoxText(0)
        return self.boxen
    def getHtml(self):
        self.html=self.tess.GetHOCRText(0)
        return self.html
    def getImage(self):
        self.parser = TessHTMLParser(self.image)
        self.parser.feed(self.html)
        self.image= self.parser.getImage()

# class for video comparisons
class FrameEngine:
    lastFrame = None
    scrollImage = None
    frameChangeThreshold = 8.0
    blankThreshold = 10.0
    matchTemplateThreshold =  5000000.0

    def __init__ (self):
        self.clearScrollImage()
        return
        
    ''' implement a faster algorithm
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
        ''' compare grayscale histogram to that of a typical WinXP desktop'''
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
            self.lastFrame = vframe
            logging.info( "frameChanged: First frame or starting SkimmingMode")
            return True
        if self.lastFrame.shape != vframe.shape:
            # current frame is different from lastFrame in size or color depth; reset last frame
            self.lastFrame = vframe
            logging.info( "frameChanged: Error: lastFrame of different size")
            return True
        diffFrame = cv2.absdiff(vframe, self.lastFrame)
        #debug
        #print (str(self.frameChangeThreshold)+" - "+str( np.mean(diffFrame)))
        
        self.lastFrame = vframe # update frame
        if (np.mean(diffFrame) <self.frameChangeThreshold):
            #logging.info( "frameChanged: Change not big enough" +str( np.mean(diffFrame)))
            return False
        else:
            logging.info( "frameChanged: Changed " + str( np.mean(diffFrame)))
            return True

    def clearScrollImage (self):
        self.scrollImage = None
    def clearLastFrame (self):
        self.lastFrame = None
            
    def setScrollImage (self, img):
        self.scrollImage = np.copy(img)
            
    def getScrollImage (self):
        return self.scrollImage 
                            
    def featureMatch (self, template, r_threshold = 0.6):
        ''' use SURF/FLANN to map the template with an internal scrollImage, return the offset of template
        inside the scrollImage, plus the features; None if something goes wrong'''
        #import numpy as np
        #import cv2
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
        logging.info('featureMatch: image - '+str(len(kp1))+' features, template - '+str(len(kp2))+' features')
        
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
        logging.info( 'featureMatch: %d / %d  inliers/matched' % (np.sum(status), len(status)))

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

            
    def findMatch (self, image, template, expectedLoc=(0,0)):
        '''find template in image; return (location, minV) if found, or None is not found'''
        minV,maxV,minL ,maxL = None, None, None, None
        if not len(image.shape)==len(template.shape):
            # different color depth
            logging.error("findMatch: image and template are of different color depths")
            return None
        try:
            res = cv2.matchTemplate(image,template,cv.CV_TM_SQDIFF)
            minV,maxV,minL ,maxL = cv2.minMaxLoc(res)
        except:
            # most likely the images have different format
            logging.error("findMatch: matchTemplate failed")
            return None
        if minL is None:
            #logging.info("findMatch: matchTemplate returns nothing")
            return None
        
        #logging.info("findMatch: "+str(minL)+", "+str(res[minL[1], minL[0]])+ " vs "+str(expectedLoc)+", "+ str(res[expectedLoc[1], expectedLoc[0]]))
        # sometimes minMaxLoc misses the min when matching texts, probably because the gradient is not smooth 
        #if (res[minL[1], minL[0]] > res[expectedLoc[1], expectedLoc[0]]):
                # the original location is the min
                #logging.info("findMatch: minLoc is not the real minimal, compared to the expectedLoc")
        #        return expectedLoc
        if (minV<self.matchTemplateThreshold):
            # ignore small differences
            #logging.info("findMatch: found match"+str(minL))
            return minL, minV
        else:
            #logging.info("findMatch: no match found; minVal is too large, mostly like a mismatch")
            return None

    
##################
# functions
def onChange (c):
    video.set(cv.CV_CAP_PROP_POS_FRAMES, c*100)

aoilist=[]
def updateAOI (data):
    ''' This function takes a tuple with 7 elements and append it to the global aoilist[].
    Data: (PageTitle, aoiID, aoiContent, x1, y1, x2, y2)
    '''
    global aoilist
    
    if type(data)!=tuple or len(data)!=7:
        print "Error in UpdateAOI: data = "+str(data)
        return
        
    aoilist.append(data)
        
# funcs to process the YAML config file
signatureImageDict={}
def p2ReadSignatureImage(k, fname, c):
    '''Takes a key, a value (file name), and a context, and reads the image if key="match"
    then updates the global dict signatureImageDict'''
    global signatureImageDict
    
    
    # set colorplane choices
    colorPlane = -99; #use all colors
    if "useGrayscaleImage" in yamlconfig["study"].keys() and yamlconfig["study"]["useGrayscaleImage"]==True:
        colorPlane = -1
    elif "useColorChannel" in yamlconfig["study"].keys():
        if yamlconfig["study"]["useColorChannel"] == "B":
            colorPlane = 0
        elif yamlconfig["study"]["useColorChannel"] == "G":
            colorPlane = 1
        elif yamlconfig["study"]["useColorChannel"] == "R":
            colorPlane = 2
        else:
            colorPlane = -1
    logging.info("ColorPlane = "+str(colorPlane))

    
    # now get the last key, and if it's not "match" then return True to move to the next node
    if not k=="match": 
        return True
    img = None
    if "imgFilePath" in yamlconfig["study"].keys():
        fname = os.path.join(yamlconfig["study"]["imgFilePath"], fname)
    try:
        img = cv2.imread(fname)
        logging.info("p2ReadSignatureImage: reading image file="+str(fname))
    except:
        logging.error("Error p2ReadSignatureImage: error reading image file="+str(fname))
        return True
    # convert frame to single channel if needed
    if len(img.shape)>2 and colorPlane == -1:
        # grayscale
        img = cv2.cvtColor(img, cv.CV_RGB2GRAY)
    elif len(img.shape)>2 and 0<=colorPlane<=2:
        # color plane
        img = cv2.split(img)[colorPlane]
    else:
        # full color
        pass

    signatureImageDict[fname]=img
    return True
                
def p2Task(k, value, context):
    '''A callback function for p2YAML(). It taks a list of keys (context) and a Value from the
    yamlconfig, and takes appropriate actions; 
    returns 
        None if no-match, ==> stop processing any subnodes
        True if we want to continue processing the rest of the sibling elements ; 
        False to stop processing sibling elements
    '''

    global signatureImageDict, frame, txt, yamlconfig, skimmingMode
    
    #print "p2Task: k="+str(k) +" v="+str(v)
    # need to look into the v for a field called "match"
    if not isinstance(value, dict):
        # not a dict, no need to process
        return True
    # check if there is a field "match"
    if "match" in value:
        # first make sure v is in the signature image list
        fname = value["match"]
        # if image path name is specified, can be absolute or relative
        if "imgFilePath" in yamlconfig["study"].keys():
            fname = os.path.join(yamlconfig["study"]["imgFilePath"], fname)
        
        if not (fname in signatureImageDict):
            logging.error("SignatureMatch: context="+str(context)+" fname="+str(fname)+" is not in the SignatureImageDict"+txt)
            return True
        res = frameEngine.findMatch(frame, signatureImageDict[fname])
        if res is None:
            # no match found; stop processing child nodes
            #logging.info("SignatureMatch: context="+str(context)+" fname="+str(fname)+" is not found in the current frame")
            return None
        # if match fails, move to the next match; only proceed if Match succeeded
        else:
            # found match, break; print "==== Match! ==" + fname
            taskSigLoc, minVal=res
            logging.info("MATCH: Signature="+str(fname)+"\tLocation="+str(taskSigLoc)+"\tminVal="+str(minVal)+txt)
            
    # if successful match or NO match needed
    if "log" in value: 
        # simply log the message
        logging.info("LOG: context="+str(context)+"\tmsg="+value["log"]+txt)
    if "aoi" in value:
        # an AOI defined directly by coordinates; will output and add the aoi for matching
        coord = map(int, value["aoi"].split(","))   # by default, in order x1, y1, x2, y2
        if "aoiFormat" in yamlconfig["study"]:
            if yamlconfig["study"]["aoiFormat"] == "xywh":
                # the x,y,w,h format: convert to xyxy format
                coord[2]=coord[2]+coord[0]
                coord[3]=coord[3]+coord[1]
        pageTitle = "/".join(context)        # 'Assessment/items/Task3DearEditor/tab1', only path to the parent 
        logging.info("AOI: pageTitle="+pageTitle+"\tid='"+str(k)+"'\tbox="+str(coord)+"\tcontent='"+str(k)+"'")
        updateAOI((pageTitle, str(k), str(k), coord[0], coord[1], coord[2], coord[3]))
    
    if "ocr" in value: 
        coord = map(int, value["ocr"].split(","))   # in order x1, y1, x2, y2
        if "aoiFormat" in yamlconfig["study"]:
            if yamlconfig["study"]["aoiFormat"] == "xywh":
                # the x,y,w,h format: convert to xyxy format
                coord[2]=coord[2]+coord[0]
                coord[3]=coord[3]+coord[1]
        try:
            # in numpy, the order goes (y1:y2, x1:x2)
            if len(frame.shape)==2:
                ocrBitmap=np.copy(frame[coord[1]:coord[3], coord[0]:coord[2]])  # grayscale already
            else:
                ocrBitmap=cv2.cvtColor(np.copy(frame[coord[1]:coord[3], coord[0]:coord[2]]), cv.CV_RGB2GRAY)
        except:
            logging.error("Error getting ocrBitmap. Check YAML ocr lines. Key="+str(k)+" value="+str(value)+txt)
            return True
        try:
            ocrtext=tess.image2txt(ocrBitmap).replace("\n", " ")
        except:
            logging.error("Error doing OCR. Key="+str(k)+" value="+str(value)+txt)
            return True
        # log the text values
        logging.info("OCR: context="+str(context)+"\tcoord="+str(coord)+"\tconfidence="+str(tess.confidence)+"\ttext='"+ ocrtext[:15]+"'" +txt)
        html=tess.getHtml() # log the HTML values
        # logging OCR results?
        if "ocrLogText" in yamlconfig["study"] and yamlconfig["study"]["ocrLogText"]:
            logging.info("\nOCR TEXT BEGIN:\n"+ocrtext+"\nOCR TEXT END:\n")
        if "ocrLogHTML" in yamlconfig["study"] and yamlconfig["study"]["ocrLogHTML"]:
            logging.info("\nAOI BEGIN:\n"+html+"\nAOI END:\n")
        # set the ocr offset and pagetitle and export data
        # see if we need to export AOI
        if "outputAOI" in yamlconfig["study"] and yamlconfig["study"]["outputAOI"]:
            parser.ocrOffsetX = coord[0]; parser.ocrOffsetY = coord[1]; 
            parser.ocrPageTitle = "/".join(context+[k]) #'Assessment/items/Task3DearEditor/tab1'
            parser.callbackfunc = updateAOI #callback func to construct AOI
            logging.info("")
            parser.feed(html)  
            logging.info("")
            
    if "screenshot" in value:
        # save a screenshot
        fname = txt
        fname = fname.replace("\t","_").replace("=","_").replace(".avi","")
        fname = str(k)+fname+".png"
        fname = fname.replace("\t","").replace("'","").replace("video_","")
        print fname
        try:
            cv2.imwrite(fname, frame)
        except:
            logging.error("Error writing the current frame as a screenshot to file ="+str(fname))
        logging.info("Screenshot f='"+fname+"'"+txt)
    if "break" in value:
        # skip the rest of the tests in the same level
        if  value["break"]:
            return False
    # if we haven't returned False by this point, continue to process the sub nodes
    return True
                
def p2YAML(d, func, context=[]):
    '''This function loops through all branches of a (nested) dict or a list of (nested) dict, and 
    runs func(key, value) for all terminal nodes, i.e., dict pairs with no more embedding.
    The "context" parameter is a list to keep track of the context; context[-1] is the last key. 
    '''
    if context is None:
        context=["Root"]
    if isinstance(d, dict):
        for k,v in d.items():
            # if it's a dict, add the context and process the value
            res=func(k, v, context)
            if res is None:
                # no match, stop processing all subnodes, move to sibling nodes
                pass 
            elif res == False:
                # task returns False, stop processing sibling nodes
                break
            elif res==True:
                p2YAML(v, func, context +[k]) # or else we will process the elements
            else:
                p2YAML(v, func, context +[k])
    
def processVideo(v):
    '''Process a video file, and log the events and AOIs to the log file.
    It uses the global settings from the yamlconfig object. 
    '''
    global yamlconfig, gaze, gazex, gazey, aoilist, toffset
    global video, frame, taskSigLoc, minVal, startFrame
    global txt, essayID, lastEssayID, vTime, jumpAhead, skimmingMode
    
    # init vars
    try:
        startFrame= yamlconfig["study"]["startFrame"]
    except:
        startFrame=1
    try:
        ratio = yamlconfig["study"]["scalingRatio"]
    except:
        ratio = 1
    
    # create new log for the file v
    logfilename = os.getcwd()+"\\"+str(v)+"_AOI.log"
    logging.basicConfig(filename=logfilename, format='%(message)s', level=logging.DEBUG)
    print("OpenLogger "+logfilename)
    logging.info("\n\n======================================================")

    # close all windows, then open this one
    cv2.destroyAllWindows()
    windowName=v
    if showVideo: cv2.namedWindow(v)
    # get the video
    try:
        video = cv2.VideoCapture(v)
    except:
        logging.error("Error: cannot open video file "+v)
        print "Error: cannot open video file "+v
        return
    # getting video parameters
    nFrames = int( video.get(cv.CV_CAP_PROP_FRAME_COUNT ))
    fps = video.get( cv.CV_CAP_PROP_FPS )
    if fps<=0:
        logging.error("Error: fps=0. Bad video file="+v)
        return
    taskbarName="Video"
    if showVideo: cv2.createTrackbar(taskbarName, windowName, int(startFrame/100), int(nFrames/100+1), onChange)
    # log
    logging.info("video = "+str(v)+"\tScaling ratio =" +str(ratio) +"\tlog = '"+str(logfilename)+"'")
    logging.info("VideoFrameSize = "+str(video.get(cv.CV_CAP_PROP_FRAME_WIDTH ))+"\t"+str(video.get(cv.CV_CAP_PROP_FRAME_HEIGHT )))
    logging.info("NumFrames = "+str(nFrames)+"\tStartFrame = "+str(startFrame)+ "\tFPS="+str(fps))

    # read signature image files for template matching
    p2YAML(yamlconfig["tasks"], p2ReadSignatureImage)
    
    # read eye event logs 
    basename = os.path.basename(os.path.splitext(v)[0])
    if "logFileSuffix" in yamlconfig["study"].keys():
        datalogfilename = basename + yamlconfig["study"]["logFileSuffix"]
    else:
        datalogfilename = basename + "_eye.log"; #default
    datafilename = basename + "_events.txt"
    print "datalogfilename="+datalogfilename
    # check to see if it's empty; if so, delete it
    if os.path.isfile(datafilename) and os.path.getsize(datafilename)==0:
        print('Eyelog2Dat: %s file is empty. Deleted' % datafilename)
        os.remove(datafilename)
    if not os.access(datafilename, os.R_OK):
        print('Eyelog2Dat: %s file is not present' % datafilename)
        # now try to process and generate the file:
        if "gazeProcessingScript" in yamlconfig["study"].keys():
            awkfile= yamlconfig["study"]["gazeProcessingScript"]
        else:
            # default
            awkfile = "eyelog2dat.awk"
        f = open(datafilename, "w")
        call = ["gawk", "-f", awkfile , datalogfilename]
        logging.info("AWK: \t"+str(call))
        res = subprocess.call(call, shell=False, stdout=f)
        f.close()
        if res != 0:
            print("Error calling eyelog2dat.awk!")
            logging.error("Error calling eyelog2dat.awk!")
            # sys.exit(1)
    # read gaze
    gaze=None
    try:
        gaze = np.genfromtxt(datafilename, delimiter='\t', dtype=None, names=['t', 'event', 'x', 'y', 'info'])
    except:
        # no gaze file to read; fake one
        print "Error reading "+datafilename
        gaze = np.genfromtxt("fake_events.txt", delimiter='\t', dtype=None, names=['t', 'event', 'x', 'y', 'info'])
        
    gaze = gaze.view(np.recarray)    # now you can refer to gaze.x[100]
    
    mouse= gaze[np.where(gaze.event=="mouseClick")]
    print "mouse data len = "+str(len(mouse))
    gaze = gaze[np.where(gaze.event=="gaze")]
    print "gaze data len = "+str(len(gaze))
    
    #gaze = [row for row in reader(datafilename, delimiter='\t') if row[1] == 'gaze']
    if(gaze is not None and len(gaze) < 1):
        print("Error reading gaze data! File="+datalogfilename)
        logging.error("Error reading gaze data! File="+datalogfilename)
    print "Gaze read from "+datafilename +" with n="+str(len(gaze))
    # end reading eye event log
    
    # init
    essayID=None; lastEssayID="falseID"; #forcedCalc=False
    #gazeline = None; gazecount=0; 
    gazex=-999; gazey=-999; gazetime =0; lastGazetime=-999
    # set the flag for skimmingMode
    skimmingMode=True; frameChanged=False; skimFrames = int(jumpAhead * fps)
    aoilist=[]; dump=[]; lastCounter=0
    
    # set colorplane choices
    colorPlane = -99; #use all colors
    if "useGrayscaleImage" in yamlconfig["study"].keys() and yamlconfig["study"]["useGrayscaleImage"]==True:
        colorPlane = -1
    elif "useColorChannel" in yamlconfig["study"].keys():
        if yamlconfig["study"]["useColorChannel"] == "B":
            colorPlane = 0
        elif yamlconfig["study"]["useColorChannel"] == "G":
            colorPlane = 1
        elif yamlconfig["study"]["useColorChannel"] == "R":
            colorPlane = 2
        else:
            colorPlane = -1
    logging.info("ColorPlane = "+str(colorPlane))
    # now loop through the frames
    while video.grab():
        frameNum = video.get(cv.CV_CAP_PROP_POS_FRAMES)
        # lastCounter tracks the greedy jumpahead position, which should be within skimFrames
        # when in skimmingMode, both of these should advance; this includes when the user jumps ahead with the slider
        if lastCounter<frameNum: lastCounter = frameNum
        # if lastCounter is way ahead of frameNum, it's clear that it's caused by user rewind; reset
        if lastCounter - frameNum > skimFrames+10: lastCounter = frameNum
        # if in the refined search mode and the frameNum catches with lastCounter, then we resume skimming
        if not skimmingMode and frameNum == lastCounter :
            # not in skimmingMode but we have scanned all the frames in between
            skimmingMode = True
            
        # skipping frames at a time
        if skimmingMode and frameNum % skimFrames >0 : continue
        
        # read the frame
        flag, frame = video.retrieve()
        
        # convert frame to single channel if needed
        if len(frame.shape)>2 and colorPlane == -1:
            # grayscale
            frame = cv2.cvtColor(frame, cv.CV_RGB2GRAY)
        elif len(frame.shape)>2 and 0<=colorPlane<=2:
            # color plane
            frame = cv2.split(frame)[colorPlane]
        else:
            # full color
            pass
            
        if flag:
            # captions
            vTime = video.get(cv.CV_CAP_PROP_POS_MSEC)
            #frameNum = video.get(cv.CV_CAP_PROP_POS_FRAMES)
            txt="\tvideo='"+v+"'\tt="+str(vTime) +'\tframe='+str(frameNum)

            ##############################
            # mouse click logging
            ##############################
            if (mouse is not None and len(mouse)>1):
                temp = mouse[np.where(mouse.t<=vTime+toffset)]   
                temp = temp[np.where(temp.t>lastGazetime)]   
                #print "mouse = "+str(len(temp))
            #@ need to export all mouse events since last time, or skipping frame will skip mouse events
                if len(temp)>0:
                    for i in temp:
                        # found at least 1 match
                        gazetime= i["t"]
                        gazex=int(i["x"])
                        gazey=int(i["y"])
                        if (not lastGazetime ==gazetime):
                            logging.info("MouseClick: vt="+str(vTime)+"\tgzt="+str(gazetime)+"\tx="+str(gazex)+"\ty="+str(gazey))
                        lastGazetime=gazetime

            ################################################
            # now only process when there is a large change
            ################################################
            frameChanged = frameEngine.frameChanged(frame)
            # logging.info("SkimmingMode ="+str(skimmingMode)+", lastCounter= "+str(lastCounter)+" frameNum= "+str(frameNum)+" skimFrames= "+str(skimFrames))
            if (frameChanged and skimmingMode and frameNum>skimFrames):
                # now we need to rewind and redo this in the normal mode
                skimmingMode = False
                lastCounter = frameNum+1   #lastCounter tracks where we have skimmed to
                video.set(cv.CV_CAP_PROP_POS_FRAMES, frameNum-skimFrames)
                logging.info("SkimmingMode:\t"+str(skimmingMode)+"\tlastCounter="+str(lastCounter)+"\tframeNum="+str(frameNum)+"\tskimFrames= "+str(skimFrames))
                continue
                
            #if (frameEngine.frameChanged(cv2.resize(frame, (0,0), fx= 0.25, fy=0.25)) or forcedCalc): #no significant performance gain
            if (frameChanged and not skimmingMode):
                # let's do template matching to find if this is a valid task screen
                taskSigLoc, minVal=None,None; 
                # now go through the tasks and items
                aoilist = []
                p2YAML(yamlconfig["tasks"], p2Task)     # this implicitly fills the aoilist[]
                aoilist = np.array(aoilist, dtype=[('page', 'S80'), ('id', 'S20'), ('content','S80'), ('x1',int), ('y1',int), ('x2',int), ('y2',int)])
                aoilist = aoilist.view(np.recarray)
                #if len(aoilist)>0:
                #    logging.info("AOIDump\n"+str(aoilist[0])+"\nAOIDUMP END\n")
                # done with the frame

            
            ##############################
            # AOI logging
            ##############################
            # disabled
            if False and (gaze is not None and len(gaze) > 1 and len(aoilist)>1):
                # @@ this is where we should recalibrate 
                # (a) redefine lineHeight and wordWidth so that no gap is left
                # (b) get heatmap and estimate distribution to the left and top
                  # edges and other "good features"; calc teh best fit
                # (c) does kernalDensity or bleeding, so that we get a matrix
                  # of the "activation" on each AOI over time
                temp = gaze[np.where(gaze.t<=vTime+toffset)]   
                if len(temp)>0:
                    # found at least 1 match
                    gazetime= temp.t[-1]
                    gazex=int(temp.x[-1])
                    gazey=int(temp.y[-1])
                # now need to find the AOI and log it
                # this means that the p2Task() need to have a global AOI array
                if not np.isnan(gazex+gazey)  and gazetime !=lastGazetime:
                    dump=aoilist[np.where(aoilist.x1<=gazex )]
                    dump=dump[np.where(dump.x2>gazex)]
                    dump=dump[np.where(dump.y1<=gazey)]
                    dump=dump[np.where(dump.y2>gazey)]
                    if len(dump)>0:
                        for aoi in dump:
                            logging.info("Gaze: vt="+str(vTime)+"\tgzt="+str(gazetime)+"\tx="+str(gazex)+"\ty="+str(gazey)+"\taoi="+"\t".join([str(s) for s in aoi]))
                    else:
                        # gaze is not on an AOI
                        logging.info("Gaze: vt="+str(vTime)+"\tgzt="+str(gazetime)+"\tx="+str(gazex)+"\ty="+str(gazey)+"\taoi=")
                    lastGazetime=gazetime     
            # end of AOI
            ############################
            # display video
            ############################
            if showVideo:
                text_color = (128,128,128)
                txt = txt+"\t"+str(parser.ocrPageTitle)
                cv2.putText(frame, txt, (20,50), cv2.FONT_HERSHEY_PLAIN, 1.0, text_color, thickness=1)
                # shows the gaze circle
                if not np.isnan(gazex+gazey): cv2.circle(frame, (int(gazex), int(gazey)), 20, text_color)
                # displays the AOI of the last matched object
                if len(dump)>0: 
                    for d in dump:
                        cv2.rectangle(frame, (d["x1"], d["y1"]), (d["x2"], d["y2"]) ,text_color, 2)
                    cv2.rectangle(frame, (dump.x1[-1], dump.y1[-1]), (dump.x2[-1], dump.y2[-1]) ,text_color,2)
                # now show mouse, last pos; used to estimate toffset
                #curmouse = mouse[np.where(mouse.t<=vTime+ toffset)]
                curmouse = gaze[np.where(gaze.t<=vTime+ toffset)]
                if curmouse is not None and len(curmouse)>1: 
                    cv2.circle(frame, (int(curmouse.x[-1]), int(curmouse.y[-1])), 10, text_color, -1)
                    
                cv2.imshow(windowName, frame)       # main window with video control
                #if txtScrollImage is not None: cv2.imshow("txtScrollImage", txtScrollImage)
                #if txtBitmap is not None: cv2.imshow("txt", txtBitmap)
                    
                # keyboard control; ESC = quit
                key= cv2.waitKey(1) #key= cv2.waitKey(waitPerFrameInMillisec)
                if (key==27):
                    logging.info("Key: ESC pressed"+txt)
                    break
                elif (key==32):
                    logging.info("Key: paused"+txt)
                    cv2.waitKey()
                # elif (key==43):
                    # frameNum+=500
                    # video.set(cv.CV_CAP_PROP_POS_FRAMES, frameNum)
                    # forcedCalc=True
                # elif (key==45):
                    # frameNum-=500
                    # video.set(cv.CV_CAP_PROP_POS_FRAMES, frameNum)
                    # forcedCalc=True
                elif (key>0):
                    # any other key saves a screenshot of the current frame
                    logging.info("Key: key="+str(key)+"\tvideoFrame written to="+v+"_"+str(essayID)+"_"+str(vTime)+".png"+txt)
                    cv2.imwrite(v+"_"+str(essayID)+"_"+str(vTime)+".png", frame)
                    print key
                else: pass

            # console output, every 200 frames        
            if (frameNum%1000 ==0):
                print " "+str(int(vTime/1000)), 
                if showVideo: cv2.setTrackbarPos(taskbarName, windowName, int(frameNum/100))
        else:
            logging.error("Error reading video frame: vt="+str(vTime)+"\tgzt="+str(gazetime)+"\tx="+str(gazex)+"\ty="+str(gazey)+"\taoi=")
            pass # no valid video frames; most likely EOF

    logging.info("Ended:"+txt)
    logging.shutdown()    

if __name__ == "__main__":

    ##########################
    # global/main:
    ###########################################################
    # Starting
    showVideo=True
    v="*.avi"
    # video input filename
    if(len(sys.argv)==3):
        yamlfile = sys.argv[1]
        v=sys.argv[2]
    elif(len(sys.argv)==4):
        yamlfile = sys.argv[1]
        v=sys.argv[2]
        if(sys.argv[3]=="-novideo"): showVideo=False
    else:
        print "Usage: python cv2video.py config.yaml videofile.avi [-novideo]"
        print "where: config.yaml is the configuration file for the video"
        print "       videofile.avi is the video file to process"
        print "       and we assume the videofile_eye.log file is in the same directory"
        print "       in order to extract the eyegaze data"
        print "       Use -novideo to speed up the processing by x2 or more"
        sys.exit(0)
    
    # getting yaml definition and make sure all the required elements are there
    yamlfile = sys.argv[1]
    yamlconfig = yaml.load(open(yamlfile))
    assert "tasks" in yamlconfig
    assert "study" in yamlconfig
    #assert "Assessment" in yamlconfig["tasks"]
    #assert "items" in yamlconfig["tasks"]["Assessment"]

    # ORC engine
    tess = TessEngine()
    # instantiate the parser and fed it some HTML
    parser = TessHTMLParser()
    parser.ocrZoomRatio = tess.ocrZoomRatio

  
    # Frame engine
    frameEngine = FrameEngine()
    # gary feng: for CBAL tabs
    frameEngine.frameChangeThreshold=yamlconfig["study"]["frameChangeThreshold"]

        
    #use logger
    #logfilename =os.path.dirname(sys.argv[0])+"\\cv2video.log"
    #logging.basicConfig(filename=logfilename, format='%(message)s', level=logging.DEBUG)
    #print("OpenLogger "+logfilename)
    #logging.info("==============================================================\n\n")
    #logging.info("OpenCV version = "+str(cv2.__version__))


    # init vars
    startFrame= yamlconfig["study"]["startFrame"]
    #forcedCalc=False
    lastFrame=None
    diffFrame=None
    tmp=None; match=None; tmp_old=None
    halfFrame=None
    halfImage=None
    minLoc = None
    minVal = None
    taskSigLoc=None
    txtBitmap=None
    txtScrollImage=None
    txt=""
    video=None

    frame=None
    vTime=0
    essayID=None
    lastEssayID="falseID"
    gaze=None
    gazex=0; gazey=0;
    # for skimmingMode, # of seconds to jump ahead
    if "jumpAhead" in yamlconfig["study"]:
        jumpAhead = yamlconfig["study"]["jumpAhead"]
    else:
        jumpAhead = 0.5
    skimmingMode=False
    
    # this is the async estimated by looking at video mouse movement and 
    #  cursor display based on the data from the mouse event log
    # quick hack, should be estimated automatically using template matching
    if "videogazeoffset" in yamlconfig["study"]:
        toffset = yamlconfig["study"]["videogazeoffset"]
    else:
        toffset = -600
    

    #logging.info("VideoFilePattern = "+str(v))
    videoFileList = glob.glob(v)
    #logging.info("VideoFileList = "+str(videoFileList))

    # process videos
    for vf in videoFileList:
        #logging.info("Processing video = "+str(v))
        processVideo(vf)
                
    # done
    if showVideo: cv2.destroyAllWindows()
    #logging.info("End")
    logging.shutdown()
    print "end processing"

    # sys.exit(0)
