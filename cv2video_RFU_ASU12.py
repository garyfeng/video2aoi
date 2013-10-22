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

import tesseract
from HTMLParser import HTMLParser
from htmlentitydefs import name2codepoint

# create a subclass and override the handler methods
class TessHTMLParser(HTMLParser):
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
        pass

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
            
    def frameChanged (self, vframe):
        '''compare the current frame to the last frame and return a boolean'''
        if self.lastFrame is None:
            # lastFrame not set
            self.lastFrame = vframe
            logging.info( "frameChanged: First frame")
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
        
        # start feature extraction
        surf = cv2.SURF(1000)
        kp1, desc1 = surf.detectAndCompute(template, None)
        #kp2, desc2 = surf.detectAndCompute(image, None)
        kp2, desc2 = surf.detectAndCompute(self.scrollImage, None)
        if desc1 is None: 
            logging.error("featureMatch: error no feature can be found for template")
            return None
        if desc2 is None: 
            logging.error("featureMatch: error no feature can be found for self.scrollImage")
            # if the scrollImage is blank, we need to get somethings to start with. copy the template
            if desc1 is not None: self.scrollImage = np.copy(template)
            else: self.scrollImage=None
            return None
       
        desc1.shape = (-1, surf.descriptorSize())
        desc2.shape = (-1, surf.descriptorSize())
        logging.info('featureMatch: image - %d features, template - %d features' % (len(kp1), len(kp2)))
        
        # FLANN
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
    global counter, video, forcedCalc
    counter=c*100
    video.set(cv.CV_CAP_PROP_POS_FRAMES, counter)
    #forcedCalc=True
        
def processTask(taskName):
    global frame, txt, essayID, lastEssayID, frameEngine, txtBitmap, txtScrollImage, minLoc, vTime
    
    # first see if essayID has changed
    # check for essay==>non-essay transitions, dump the scroll image
    if (essayID != lastEssayID) & (essayID is None) & (lastEssayID != "falseID"):
        # to check if the file exists
        txtScrollImage=frameEngine.getScrollImage()
        # OCR scrollImage
        tess.image2txt(txtScrollImage)
        html=tess.getHtml()
        logging.info("AOI BEGIN:\tessay="+str(lastEssayID)+"\tlocation=" + str(textarea))
        logging.info(html)
        logging.info("AOI END:\tessay="+str(lastEssayID)+"\tlocation=" + str(textarea))
        #boxen=tess.getBoxen()

        # save scrollImage
        c=1
        scrollImgFileName=str(lastEssayID)+"_scroll_"+str(c)+".png"
        while os.path.isfile(scrollImgFileName):
            scrollImgFileName=str(lastEssayID)+"_scroll_"+str(c)+".png"
            c+=1
        cv2.imwrite(scrollImgFileName, txtScrollImage)
        logging.info("Essay Ended:\tessay="+str(lastEssayID)+"\tScrollingImage="+scrollImgFileName+"\tt="+str(vTime))
        # this is critical
        lastEssayID=essayID
    
    if taskName is None:
        logging.info("ProcessingTask: taskName==None")
    elif taskName=="desktop":
        logging.info("ProcessingTask: taskName==\"desktop\"")
    elif frameEngine.isDesktop(frame):
        logging.info("ProcessingTask: taskName==\"desktop\" by: isDesktop()")        
    elif taskName=="calibration":
        logging.info("ProcessingTask: taskName==\"calibration\"")
    elif taskName=="controlpanel":
        logging.info("ProcessingTask: taskName==\"controlpanel\"")
    elif taskName=="essay":
        logging.info("ProcessingTask: taskName==\"essay\"")
        # we get a grayscale of the CR image from the ORIGINAL frame (not the half frame)
        crBitmap=cv2.cvtColor(np.copy(frame[cr[0][1]:cr[1][1], cr[0][0]:cr[1][0]]), cv.CV_RGB2GRAY)
        essayID=tess.image2txt(crBitmap).replace("\n", "")
        logging.info("ProcessingTask: essayID=="+essayID)
        essayID=essayID.replace("CR No: ", "")
        txt+= "\tessayID="+essayID
        if (essayID != lastEssayID) & (lastEssayID is not None):
            # new essay starts
            # to check if the file exists
            if (lastEssayID != "falseID"):
                txtScrollImage=frameEngine.getScrollImage()
                # OCR scrollImage
                tess.image2txt(txtScrollImage)
                html=tess.getHtml()
                logging.info("AOI BEGIN:\tessay="+str(lastEssayID)+"\tlocation=" + str(textarea))
                logging.info(html)
                logging.info("AOI END:\tessay="+str(lastEssayID)+"\tlocation=" + str(textarea))
                #boxen=tess.getBoxen()

                # save scrollImage
                c=1
                scrollImgFileName=str(lastEssayID)+"_scroll_"+str(c)+".png"
                while os.path.isfile(scrollImgFileName):
                    scrollImgFileName=str(lastEssayID)+"_scroll_"+str(c)+".png"
                    c+=1
                cv2.imwrite(scrollImgFileName, txtScrollImage)
                logging.info("Essay Ended:\tessay="+str(lastEssayID)+"\tScrollingImage="+scrollImgFileName+"\tt="+str(vTime))
            c=1
            scrollImgFileName=str(essayID)+"_scroll_"+str(c)+".png"
            while os.path.isfile(scrollImgFileName):
                scrollImgFileName=str(essayID)+"_scroll_"+str(c)+".png"
                c+=1
            txt+= "\tscrollImgFileName="+scrollImgFileName
            logging.info("New Essay Started:\t"+txt)
            print "New Essay Started:\t"+txt
            txtScrollImage=None
            frameEngine.clearScrollImage()
            logging.info("frameEngine.clearScrollImage()")
            # this is critical
            lastEssayID=essayID

        # get txt image
        txtBitmap=cv2.cvtColor(np.copy(frame[textarea[0][1]:textarea[1][1], textarea[0][0]:textarea[1][0]]), cv.CV_RGB2GRAY)
        # check to see if the txtBitmap is blank or part of the desktop
        # this may happen when IE is in the transition of loading the textarea, in which case we should not update the ScrollImage
        if frameEngine.isDesktop(txtBitmap): 
            logging.info("ProcessingTask: isDesktop(txtBitmap) == True")
            pass
        elif frameEngine.isBlank(txtBitmap): 
            logging.info("ProcessingTask: isBlank(txtBitmap) == True")
            pass
        # match template to find the amount of scrolling
        elif txtScrollImage is None:
            # txtScrollImage=np.copy(txtBitmap)
            frameEngine.setScrollImage(txtBitmap)
            txtScrollImage=frameEngine.getScrollImage()
        else:
            # SURF/FLANN to find out whether there is scrolling
            logging.info("Do featureMatch, videoFrame= "+str(counter))
            res = frameEngine.featureMatch(txtBitmap)
            if res is None:
                logging.info("===> No match found")
            else:
                offset, match=res
                txtScrollImage=frameEngine.getScrollImage()
                logging.info("offset = " + str(offset))
                # scroll=False
                if (match[1]<100): 
                    logging.info("===> Too few features, n="+str(match[1]))
                elif (match[0]*1.0/match[1]<0.5): 
                    logging.info("===> Poor match, %="+str(match[0]*1.0/match[1]))
                elif (abs(offset[0])>1): logging.info("===> Mismatch, x="+str(offset[1]))
                elif (abs(offset[1])>0):
                    logging.info("===> Scroll, y="+str(offset[1])+" %="+str(match[0]*1.0/match[1]))
                    logging.info("Scroll\ty="+str(offset[1])+"\t"+txt)
                else: logging.info("No scroll")

                # update minLoc
                if offset is not None:
                    minLoc=(textarea[0][0]+offset[0], textarea[0][1]+offset[1])
    elif taskName=="banads":
        logging.info("ProcessingTask: taskName==\"banads\"")
        # we get a grayscale of the CR image from the ORIGINAL frame (not the half frame)
        crBitmap=cv2.cvtColor(np.copy(frame[cr[0][1]:cr[1][1], cr[0][0]:cr[1][0]]), cv.CV_RGB2GRAY)
        essayID=tess.image2txt(crBitmap).replace("\n", "")
        logging.info("ProcessingTask: essayID==\""+essayID+"\"")
        #essayID=essayID.replace("CR No: ", "")
        txt+= "\tessayID=\""+essayID+"\""
        if True:
            # new essay starts
            # to check if the file exists
            if (lastEssayID != "falseID"):
                #txtScrollImage=frameEngine.getScrollImage()
                # get txt image
                txtBitmap=cv2.cvtColor(np.copy(frame[textarea[0][1]:textarea[1][1], textarea[0][0]:textarea[1][0]]), cv.CV_RGB2GRAY)

                # OCR scrollImage
                #tess.image2txt(txtScrollImage)
                tess.image2txt(txtBitmap)
                html=tess.getHtml()
                logging.info("AOI BEGIN:\tessay="+str(lastEssayID)+"\tlocation=" + str(textarea))
                logging.info(html)
                logging.info("AOI END:\tessay="+str(lastEssayID)+"\tlocation=" + str(textarea))
                #boxen=tess.getBoxen()

                logging.info("Essay Ended:\tessay="+str(lastEssayID)+"\tScrollingImage="+scrollImgFileName+"\tt="+str(vTime))
            c=1
            txtScrollImage=None
            frameEngine.clearScrollImage()
            logging.info("frameEngine.clearScrollImage()")
            # this is critical
            lastEssayID=essayID

    #f.write(txt+"\n")
    logging.info("FrameChange\t"+txt)

def processVideo(v):
    global counter, video, frame, halfFrame, minLoc, minVal, taskSigLoc, txtBitmap 
    global txtScrollImage, forcedCalc, startFrame, txt, essayID, lastEssayID, vTime, cr, textarea, ratio, logfilename
    
    # init vars
    startFrame= 1
    forcedCalc=False
    #lastFrame=None
    #diffFrame=None
    #tmp=None; match=None; tmp_old=None
    halfFrame=None
    #halfImage=None
    minLoc = None
    minVal = None
    taskSigLoc=None
    txtBitmap=None
    txtScrollImage=None
    
    # create new log for the file v
    logfilename = os.path.dirname(sys.argv[0])+"\\"+str(v)+"_AOI.log"
    loglevel = logging.INFO
    logformatter = logging.Formatter("%(asctime)s\t%(message)s")
    rootlogger = logging.getLogger('')
    systemloghandler = logging.FileHandler(logfilename)
    systemloghandler.setLevel(loglevel)
    systemloghandler.setFormatter(logformatter)
    rootlogger.addHandler(systemloghandler)
    
    print("OpenLogger "+logfilename)
    logging.info("======================================================\n\n")
    logging.info("OpenCV version = "+str(cv2.__version__))
    logging.info("video = "+str(v))
    logging.info("log = "+str(logfilename))
    logging.info("Page Identifier @ location " + str(cr))
    logging.info("Text Area @ location " + str(textarea))
    logging.info("Scaling ratio =" +str(ratio))

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
    logging.info("# of frames = "+str(nFrames))
    logging.info("Starting frames = "+str(startFrame))
    # try to set it as 15fps
    fps = video.get( cv.CV_CAP_PROP_FPS )
    logging.info( "FPS="+str(fps))
    try:
        waitPerFrameInMillisec = int( 1000/fps )
    except:
        # fps=0, bad video file. Don't proceed
        logging.error("Error: Bad video file="+v)
        #sys.exit(-1)
        return

    taskbarName="Video"
    if showVideo: cv2.createTrackbar(taskbarName, windowName, int(startFrame/100), int(nFrames/100+1), onChange)

    logging.info("Video Frame size = ("+str(video.get(cv.CV_CAP_PROP_FRAME_WIDTH ))+", "+str(video.get(cv.CV_CAP_PROP_FRAME_HEIGHT ))+")")

    signatureFileList=glob.glob("signature_*.png")
    signatureImageList = [cv2.resize(cv2.cvtColor(cv2.imread(f), cv.CV_RGB2GRAY),(0,0), fx= ratio, fy=ratio ) for f in signatureFileList]
    logging.info("Signatures = "+str(signatureFileList))

    tabFileList=glob.glob("tab_*.png")
    #tabImageList = [cv2.resize(cv2.cvtColor(cv2.imread(f), cv.CV_RGB2GRAY),(0,0), fx= ratio, fy=ratio ) for f in tabFileList]
    tabImageList = [cv2.imread(f) for f in tabFileList]
    logging.info("tab = "+str(tabFileList))

    tab2FileList=glob.glob("tab2_*.png")
    tab2ImageList = [cv2.resize(cv2.cvtColor(cv2.imread(f), cv.CV_RGB2GRAY),(0,0), fx= ratio, fy=ratio ) for f in tab2FileList]
    #tab2ImageList = [cv2.imread(f), fx= ratio, fy=ratio ) for f in tab2FileList]
    logging.info("tab2 = "+str(tab2FileList))

    essayID=None
    lastEssayID="falseID"

    counter = startFrame-1
    forcedCalc=False
    while video.grab():
        counter += 1
        tabname=None                    
        tab2name=None                    
        flag, frame = video.retrieve()
        if flag:
            # resize frame
            halfFrame = cv2.resize(cv2.cvtColor(frame, cv.CV_RGB2GRAY),(0,0), fx= ratio, fy=ratio)
            # captions
            vTime = video.get(cv.CV_CAP_PROP_POS_MSEC)
            txt="video='"+v+"'\tt="+str(vTime) +'\tframe='+str(counter)+"\tnFrame="+str(nFrames)
            # now only process when there is a large change
            if (frameEngine.frameChanged(halfFrame) or forcedCalc):
                if forcedCalc: 
                    logging.info("forcedCalc "+txt)
                    forcedCalc=False
                # let's do template matching to find if this is a valid task screen
                taskSigLoc, minVal=None,None; res=None
                taskName=None
                
                for (f, sig) in zip(signatureFileList, signatureImageList):
                    res = frameEngine.findMatch(halfFrame, sig)
                    if res is not None:
                        # found match, break
                        taskSigLoc, minVal=res
                        # get task name
                        ptn=re.compile("signature_(.+)\.png")  # can be moved outside of loop
                        dump= ptn.match(f)
                        if dump is not None: taskName = dump.group(1)
                        logging.info("Signature="+str(taskName)+"\tLocation="+str(taskSigLoc)+"\tminVal="+str(minVal))
                        # break at the first found signature
                        #break 
                # try to deal with blank frames sometimes inserted in the video
                # should not interrupt the essay unless it bears the signatures of other pre-defined pages
                if taskName is not None: essayID = None
                #processTask(taskName)
                if taskName is None:
                    #logging.info("ProcessingTask: taskName==None"+"\t"+txt)
                    logging.info("STATUS:\ttaskname='"+str(taskName)+"'\tessay='"+str(lastEssayID)+"'\ttab1='"+str(tabname)+"'\ttab2='"+str(tab2name)+"'\t"+txt)
                elif taskName=="desktop":
                    #logging.info("ProcessingTask: taskName==\"desktop\""+"\t"+txt)
                    logging.info("STATUS:\ttaskname='"+str(taskName)+"'\tessay='"+str(lastEssayID)+"'\ttab1='"+str(tabname)+"'\ttab2='"+str(tab2name)+"'\t"+txt)
                elif frameEngine.isDesktop(halfFrame):
                    #logging.info("ProcessingTask: taskName==\"desktop\" by: isDesktop()"+"\t"+txt)        
                    logging.info("STATUS:\ttaskname='"+str(taskName)+"'\tessay='"+str(lastEssayID)+"'\ttab1='"+str(tabname)+"'\ttab2='"+str(tab2name)+"'\t"+txt)
                elif taskName=="calibration":
                    logging.info("STATUS:\ttaskname='"+str(taskName)+"'\tessay='"+str(lastEssayID)+"'\ttab1='"+str(tabname)+"'\ttab2='"+str(tab2name)+"'\t"+txt)
                    #logging.info("ProcessingTask: taskName==\"calibration\""+"\t"+txt)
                elif taskName=="controlpanel":
                    logging.info("STATUS:\ttaskname='"+str(taskName)+"'\tessay='"+str(lastEssayID)+"'\ttab1='"+str(tabname)+"'\ttab2='"+str(tab2name)+"'\t"+txt)
                    #logging.info("ProcessingTask: taskName==\"controlpanel\""+"\t"+txt)
                elif taskName=="banads":
                    #logging.info("ProcessingTask: taskName==\"banads\""+"\t"+txt)

                    # processing the left tab
                    tabname=None                       
                    # try to match tab files: active left tab for CBAL BanAds
                    for (f, sig) in zip(tabFileList, tabImageList):
                        # using full frame color for matching
                        res = frameEngine.findMatch(frame, sig)
                        print "testing tab = "+f
                        if res is not None:
                            # found match, break
                            taskSigLoc, minVal=res
                            # get task name
                            ptn=re.compile("tab_(.+)\.png")  # can be moved outside of loop
                            dump= ptn.match(f)
                            if dump is not None: tabname = dump.group(1)
                            logging.info("Tab="+str(tabname)+"\tLocation="+str(taskSigLoc)+"\tminVal="+str(minVal))
                            print "Tab="+str(tabname)+"\tLocation="+str(taskSigLoc)+"\tminVal="+str(minVal)
                            # break at the first found signature
                            # break 
                    if (tabname==None): 
                        logging.info("Tab= NO_TAB_FOUND\tLocation="+str(taskSigLoc)+"\tminVal="+str(minVal))
                        print "Tab= NO_TAB_FOUND\tLocation="+str(taskSigLoc)+"\tminVal="+str(minVal)

                    # the right tab                        
                    tab2name=None                    
                    # now identify the right tab
                    for (f, sig) in zip(tab2FileList, tab2ImageList):
                        res = frameEngine.findMatch(halfFrame, sig)
                        if res is not None:
                            # found match, break
                            taskSigLoc, minVal=res
                            # get task name
                            ptn=re.compile("tab2_(.+)\.png")  # can be moved outside of loop
                            dump= ptn.match(f)
                            if dump is not None: tab2name = dump.group(1)
                            logging.info("Tab2="+str(tab2name)+"\tLocation="+str(taskSigLoc)+"\tminVal="+str(minVal))
                            print "Tab2="+str(tab2name)+"\tLocation="+str(taskSigLoc)+"\tminVal="+str(minVal)
                            # break at the first found signature
                            # break 
                    if (tab2name==None): 
                        logging.info("Tab= NO_TAB_FOUND\tLocation="+str(taskSigLoc)+"\tminVal="+str(minVal))
                        print "Tab= NO_TAB_FOUND\tLocation="+str(taskSigLoc)+"\tminVal="+str(minVal)

                    # we get a grayscale of the CR image from the ORIGINAL frame (not the half frame)
                    crBitmap=cv2.cvtColor(np.copy(frame[cr[0][1]:cr[1][1], cr[0][0]:cr[1][0]]), cv.CV_RGB2GRAY)
                    essayID=tess.image2txt(crBitmap).replace("\n", "")
                    logging.info("ProcessingTask: essayID=='"+essayID+"'"+"\t"+txt)
                    logging.info("STATUS:\ttaskname='"+str(taskName)+"'\tessay='"+str(lastEssayID)+"'\ttab1='"+str(tabname)+"'\ttab2='"+str(tab2name)+"'\t"+txt)
                    
                    # do OCR only for your essay
                    if (tabname is not None and lastEssayID != "falseID"):
                        tess.image2txt(cv2.cvtColor(np.copy(frame[307:765, 342:829]), cv.CV_RGB2GRAY))
                        html=tess.getHtml()
                        logging.info("AOI BEGIN:\tTab1='"+str(tabname)+"'\tlocation=" + str(textarea)+"\t"+txt)
                        logging.info(html)
                        logging.info("AOI END:\tTab1='"+str(tabname)+"'\tlocation=" + str(textarea)+"\t"+txt)
                    # do OCR only for your essay
                    if (tab2name== "youressay" and lastEssayID != "falseID"):
                        #txtScrollImage=frameEngine.getScrollImage()
                        # get txt image
                        txtBitmap=cv2.cvtColor(np.copy(frame[textarea[0][1]:textarea[1][1], textarea[0][0]:textarea[1][0]]), cv.CV_RGB2GRAY)

                        # OCR scrollImage
                        #tess.image2txt(txtScrollImage)
                        tess.image2txt(txtBitmap)
                        html=tess.getHtml()
                        #output
                        #logging.info("STATUS:\ttaskname='"+str(taskName)+"'\tessay='"+str(lastEssayID)+"'\ttab1='"+str(tabname)+"'\ttab2='"+str(tab2name)+"'\t"+txt)
                        
                        logging.info("AOI BEGIN:\tTab2='"+str(tab2name)+"'\tlocation=" + str(textarea)+"\t"+txt)
                        logging.info(html)
                        logging.info("AOI END:\tTab2='"+str(tab2name)+"'\tlocation=" + str(textarea)+"\t"+txt)
                        #boxen=tess.getBoxen()
                    c=1
                    txtScrollImage=None
                    frameEngine.clearScrollImage()
                    #logging.info("frameEngine.clearScrollImage()")
                    # this is critical
                    lastEssayID=essayID

            if showVideo:
                text_color = (0,0,0)
                if (taskSigLoc is not None) and (sig is not None):
                    # show matching rect
                    cv2.rectangle(halfFrame,taskSigLoc,(taskSigLoc[0]+sig.shape[1],taskSigLoc[1]+sig.shape[0]),text_color,2)
                cv2.putText(halfFrame, txt, (20,50), cv2.FONT_HERSHEY_PLAIN, 1.0, text_color, thickness=1)
                cv2.imshow(windowName, halfFrame)       # main window with video control
                if txtScrollImage is not None: cv2.imshow("txtScrollImage", txtScrollImage)
                if txtBitmap is not None: cv2.imshow("txt", txtBitmap)
                    
            if showVideo:
                # keyboard control; ESC = quit
                key= cv2.waitKey(1) #key= cv2.waitKey(waitPerFrameInMillisec)
                if (key==27):
                    logging.info("Key: ESC pressed"+"\t"+txt)
                    break
                elif (key==32):
                    logging.info("Key: paused"+"\t"+txt)
                    cv2.waitKey()
                elif (key==43):
                    counter+=500
                    video.set(cv.CV_CAP_PROP_POS_FRAMES, counter)
                    forcedCalc=True
                elif (key==45):
                    counter-=500
                    video.set(cv.CV_CAP_PROP_POS_FRAMES, counter)
                    forcedCalc=True
                elif (key>0):
                    # any other key saves a screenshot of the current frame
                    logging.info("Key: key="+str(key)+"\tvideoFrame written to="+v+"_"+str(essayID)+"_"+str(vTime)+".png"+"\t"+txt)
                    cv2.imwrite(v+"_"+str(essayID)+"_"+str(vTime)+".png", frame)
                    print key
                else: pass
            # console output, every 200 frames        
            if (counter%1000 ==0):
                print "t="+str(vTime), "essayID="+str(essayID)
                if showVideo: cv2.setTrackbarPos(taskbarName, windowName, int(counter/100))
        else:
            pass # no valid video frames; most likely EOF
    # video ends, save the scroll image
    # OCR scrollImage
    if (tabname is not None and tab2name is not None):
        #output
        logging.info("STATUS:\ttaskname='"+str(taskName)+"'\tessay='"+str(lastEssayID)+"'\ttab1='"+str(tabname)+"'\ttab2='"+str(tab2name)+"'\t"+txt)
        # left tab
        tess.image2txt(cv2.cvtColor(np.copy(frame[307:765, 342:829]), cv.CV_RGB2GRAY))
        html=tess.getHtml()
        logging.info("AOI BEGIN:\tTab1='"+str(tabname)+"'\tlocation=" + str(textarea)+"\t"+txt)
        logging.info(html)
        logging.info("AOI END:\tTab1='"+str(tabname)+"'\tlocation=" + str(textarea)+"\t"+txt)
        # right tab
        tess.image2txt(cv2.cvtColor(np.copy(frame[textarea[0][1]:textarea[1][1], textarea[0][0]:textarea[1][0]]), cv.CV_RGB2GRAY))
        html=tess.getHtml()
        logging.info("AOI BEGIN:\tTab2='"+str(tab2name)+"'\tlocation=" + str(textarea)+"\t"+txt)
        logging.info(html)
        logging.info("AOI END:\tTab2='"+str(tab2name)+"'\tlocation=" + str(textarea)+"\t"+txt)

        # save scrollImage
        #c=1
        #scrollImgFileName=str(lastEssayID)+"_scroll_"+str(c)+".png"
        #while os.path.isfile(scrollImgFileName):
        #    scrollImgFileName=str(lastEssayID)+"_scroll_"+str(c)+".png"
        #    c+=1
        #cv2.imwrite(scrollImgFileName, txtScrollImage)
        #logging.info("Essay Ended:\tessay="+str(lastEssayID)+"\tScrollingImage="+scrollImgFileName+"\t"+txt)
        logging.info("Essay Ended:\t"+txt)

##########################
# global/main:
###########################################################
# Starting
print sys.argv
print len(sys.argv)
os.chdir(os.path.dirname(sys.argv[0]))

# ORC engine
tess = TessEngine()
# Frame engine
frameEngine = FrameEngine()
# gary feng: for CBAL tabs
frameEngine.frameChangeThreshold=1.2

# show video stuff
showVideo=False

v="*.avi"
# video input filename
if(len(sys.argv)==2):
    v=sys.argv[1]
elif(len(sys.argv)==3):
    v=sys.argv[1]
    if(sys.argv[2]=="-novideo"): showVideo=False
else:
    print "Usage: cv2video.py videoname.avi [-novideo]"
    sys.exit(0)
    
#use logger
logfilename =os.path.dirname(sys.argv[0])+"\\cv2video.log"
logging.basicConfig(filename=logfilename, format='%(asctime)s\t%(message)s', level=logging.DEBUG)
print("OpenLogger "+logfilename)
logging.info("==============================================================\n\n")
logging.info("OpenCV version = "+str(cv2.__version__))


# init vars
startFrame= 1
forcedCalc=False
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
counter=0
frame=None
vTime=0
essayID=None
lastEssayID="falseID"

# to OCR CR number
# cr=((344,214), (344+170, 214+52))
cr=((445,236), (445+55, 236+16))
logging.info("Page Identifier @ location " + str(cr))
# to OCR only the textarea
#textarea=((336, 279), (336+1010, 279+555))
textarea=((874, 452), (874+420, 452+347))
logging.info("Text Area @ location " + str(textarea))


# the template for the text, somewhere in the middle of the textbox
ratio=0.5
w=int(710 * ratio)
h=int(200 * ratio)
x=int(12 * ratio)
y=int(578 * ratio)
logging.info("Scaling ratio =" +str(ratio))

logging.info("VideoFilePattern = "+str(v))
videoFileList = glob.glob(v)
logging.info("VideoFileList = "+str(videoFileList))

# process videos
for v in videoFileList:
#    try:
    logging.info("Processing video = "+str(v))
    processVideo(v)
#    except:
#        logging.error("Error: Cannot open video file "+v)
        #sys.exit(-1)
            
# done
if showVideo: cv2.destroyAllWindows()
logging.info("End")
logging.shutdown()
print "end processing"

# sys.exit(0)
