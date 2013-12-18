import cv2.cv as cv
#import tesseract
import cv2
import os
import sys
import numpy as np

import TessEngine 
#from HTMLParser import HTMLParser
from htmlentitydefs import name2codepoint

# create a subclass and override the handler methods
class MyHTMLParser(TessEngine.HTMLParser):
    box=None
    word=""
    image=None
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
            cv2.rectangle(self.image,(self.box[0], self.box[1]),(self.box[2], self.box[3]),(0,128,0),1)
        else:
            #print "box is not valid"
            pass

    def handle_endtag(self, tag):
        #print "End tag  :", tag
        pass
    def handle_data(self, data):
        # print data
        if self.box is not None:
            cv2.putText(self.image, data, (self.box[0], self.box[1]), cv2.FONT_HERSHEY_PLAIN, 1.0, 0, thickness=1)
        pass
    def handle_comment(self, data):
        print "Comment  :", data
    def handle_entityref(self, name):
        c = unichr(name2codepoint[name])
        #print "Named ent:", c
        if self.box is not None:
            cv2.putText(self.image, c, (self.box[0], self.box[1]), cv2.FONT_HERSHEY_PLAIN, 1.0, 0, thickness=1)
        pass
        
    def handle_charref(self, name):
        if name.startswith('x'):
            c = unichr(int(name[1:], 16))
        else:
            c = unichr(int(name))
        #print "Num ent  :", c
        if self.box is not None:
            cv2.putText(self.image, c, (self.box[0], self.box[1]), cv2.FONT_HERSHEY_PLAIN, 1.0, 0, thickness=1)
        pass
            
    def handle_decl(self, data):
        print "Decl     :", data

def main(frameFileName, x, y, w, h, ratio, colorPlane):
    
    tess = TessEngine.TessEngine()
    
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

    ocrtext=tess.image2txt(image).replace("\n", " ")

    html=tess.getHtml() # log the HTML values

    #image=None
    print "Recognized Text = "
    print '"'+ocrtext+'"'
    print "Confidence level= {}".format(tess.getConfidence())

    #s=["title", "id", "class"]

    # instantiate the parser and fed it some HTML
    parser = MyHTMLParser()
    parser.image = image
    parser.feed(html)

    #cv.ShowImage("IplImage", bitmap)
    cv2.imshow("Boxen", image)
    cv2.waitKey()
    cv2.destroyAllWindows()

    return ocrtext

if __name__ == "__main__":
    # unit testing

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
        print "testtess.py imageFileName [x y w h] [ratio] [colorPlane]"
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

    
    main(frameFileName, x, y, w, h, ratio, colorPlane)