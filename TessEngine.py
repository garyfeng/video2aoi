"""Tesseract Engine
Gary Feng, 2013
Printceton, NJ

This module defines classes to do OCR and to handle the HTML output.

It requires 
-- Tesseract and python-tesseract.
-- OpenCV2
-- logging with root log already defined and running (for TessHTMLParser)

"""

import tesseract

from HTMLParser import HTMLParser
from htmlentitydefs import name2codepoint

import cv2
import cv2.cv as cv
import logging
import re

class TessEngine:
    tess = None
    ocrZoomRatio = 3
    boxen = None; html=None; text=None; confidence=None
    image=None; parser=None
    
    def __init__ (self):
        """ 
        Tesseract engine: create an English OCR engine with default parameters.
        It is designed to ocr screen captures, so it first magnifies the image by ocrZoomRatio (=3).
        Then does OCR on the grayscale image. 
        
        """
        
        self.tess = tesseract.TessBaseAPI()
        self.tess.Init(".","eng",tesseract.OEM_DEFAULT)

    def image2txt (self, img):
        """Takes an img (in cv2/numpy format) and OCR using Tesseract"""
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
        """ returns the recognized text """
        return self.text
    def getConfidence(self):
        """ returns the confidence score of the last ocr """
        return self.confidence
    def getBoxen(self):
        """ returns the coordinates of the OCR boxes on page 0"""
        self.boxen = self.tess.GetBoxText(0)
        return self.boxen
    def getHtml(self):
        """ returns the HTML of the recognized text on page 0"""
        self.html=self.tess.GetHOCRText(0)
        return self.html
    def getImage(self):
        """ returns the image with OCR boxes and text overlays"""
        self.parser = imgTessHTMLParser(self.image)
        self.parser.feed(self.html)
        self.image= self.parser.getImage()
        
        
# create a subclass and override the handler methods
class TessHTMLParser(HTMLParser):
    """ This HTML parser engine outputs the AOIs to the log"""

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
                logging.error( "AOI error: handle_entityref expecting a word element here")
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
                logging.error( "AOI error: handle_charref expecting a word element here")
        return
            
    def handle_decl(self, data):
        #print "Decl     :", data
        return


    def getImage(self):
        # return self.image
        return 
        

        
# create a subclass and override the handler methods
class imgTessHTMLParser(HTMLParser):
    """This HTML parser takes an image and overlay recognized text and boxes on it."""

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

