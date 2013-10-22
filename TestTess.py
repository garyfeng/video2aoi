import cv2.cv as cv
import tesseract
import cv2
import os
import sys
import numpy as np
import atexit
import glob

from HTMLParser import HTMLParser
from htmlentitydefs import name2codepoint

# how much magnification for Tesseract to do OCR, typically 3-5
ratio=3.0
# where the text is. Default the whole page
w= -1
h= -1
x= -1
y= -1
colorPlane=-1
os.chdir(os.path.dirname(sys.argv[0]))
try:
    frameFileName = sys.argv[1]
except:
    print "testtess.py frameFileName [x y w h] [colorPlane] [ratio]"
    exit(0)
    
try:
    x = int(sys.argv[2])
    y = int(sys.argv[3])
    w = int(sys.argv[4])
    h = int(sys.argv[5])
except:
    print "[x y w h] not specified; using the whole page"

try:
    colorPlane = int(sys.argv[6])
except:
    print "[colorPlane] not specified; using default= grayscale"

try:
    ratio = float(sys.argv[7])
except:
    print "[ratio] not specified; using default="  + str(ratio)

#image=cv.LoadImage("eurotext.jpg", cv.CV_LOAD_IMAGE_GRAYSCALE)
# @@@ try if we need to use a certain color plane
if 0<=colorPlane<=2:
    print "Using color plane " +str(colorPlane)
    imgPageLayout = cv2.split(cv2.imread(frameFileName))[colorPlane]
else:
    print "No valid color plane specified; use all colors. " +str(colorPlane)
    imgPageLayout = cv2.imread(frameFileName, cv.CV_LOAD_IMAGE_GRAYSCALE)

# if no x y w h is specified, use the whole image
if h<0:
    x=0; y=0;
    w=imgPageLayout.shape[0]
    h=imgPageLayout.shape[1]
    

cv2.imshow("Frame", imgPageLayout)
tmp=np.copy(imgPageLayout[y:y+h, x:x+w ])
#cv2.imshow("cut", tmp)
image = cv2.resize(tmp,(0,0), fx= ratio, fy=ratio )
#cv2.imshow("resize", image)
#print image.shape

# Nope, Tesseract doesn't want a cvMAT
#cvImage = cv.CreateMat(image.shape[0], image.shape[1], cv.CV_32FC1)
#cvImage = cv.fromarray(image)
#cv.ShowImage("CV Image", cvImage)
#print cvImage.shape

# Tesseract wants a grayscale IPLimage file. Can't use RGB or numpy array.
bitmap = cv.CreateImageHeader((image.shape[1], image.shape[0]), cv.IPL_DEPTH_8U, 1)
cv.SetData(bitmap, image.tostring(), image.dtype.itemsize * 1 * image.shape[1])

#cv.ShowImage("IplImage", bitmap)


# cvImage = cv.LoadImage("pagelayout.png", cv.CV_LOAD_IMAGE_GRAYSCALE)

api = tesseract.TessBaseAPI()
api.Init(".","eng",tesseract.OEM_DEFAULT)
#api.SetPageSegMode(tesseract.PSM_SINGLE_WORD)
api.SetPageSegMode(tesseract.PSM_AUTO)
tesseract.SetCvImage(bitmap,api)
text=api.GetUTF8Text()
conf=api.MeanTextConf()
# GetBoxText(page#)
boxen = api.GetBoxText(0)
html=api.GetHOCRText(0)
#image=None
print "Recognized Text = "
print text
print "Confidence level:" +str(conf)

s=["title", "id", "class"]
# create a subclass and override the handler methods
class MyHTMLParser(HTMLParser):
    box=None
    word=""
    def handle_starttag(self, tag, attrs):
        # print "Start tag:", tag
        #self.box = None
        for attr in attrs:
            # print "     attr:", attr
            # if (attr[0] in s): print attr[0]+": "+ attr[1]
            if (attr[0]=="title"):
                b=attr[1].split(" ")
                if (b[0]!="bbox"): continue
                try:
                    self.box = [int(x) for x in b[1:]]
                except:
                    self.box=None
        # now draw box
        if self.box is not None:
            cv2.rectangle(image,(self.box[0], self.box[1]),(self.box[2], self.box[3]),(0,128,0),1)
        else:
            #print "box is not valid"
            pass

    def handle_endtag(self, tag):
        #print "End tag  :", tag
        pass
    def handle_data(self, data):
        # print data
        if self.box is not None:
            cv2.putText(image, data, (self.box[0], self.box[1]), cv2.FONT_HERSHEY_PLAIN, 1.0, 0, thickness=1)
        pass
    def handle_comment(self, data):
        print "Comment  :", data
    def handle_entityref(self, name):
        c = unichr(name2codepoint[name])
        #print "Named ent:", c
        if self.box is not None:
            cv2.putText(image, c, (self.box[0], self.box[1]), cv2.FONT_HERSHEY_PLAIN, 1.0, 0, thickness=1)
        pass
        
    def handle_charref(self, name):
        if name.startswith('x'):
            c = unichr(int(name[1:], 16))
        else:
            c = unichr(int(name))
        #print "Num ent  :", c
        if self.box is not None:
            cv2.putText(image, c, (self.box[0], self.box[1]), cv2.FONT_HERSHEY_PLAIN, 1.0, 0, thickness=1)
        pass
            
    def handle_decl(self, data):
        print "Decl     :", data

# instantiate the parser and fed it some HTML
parser = MyHTMLParser()
parser.feed(html)

#cv.ShowImage("IplImage", bitmap)
cv2.imshow("Boxen", image)
cv2.waitKey()
cv2.destroyAllWindows()

