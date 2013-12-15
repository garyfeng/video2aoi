import numpy as np
import logging
import cv2
import cv2.cv as cv
import os
import os.path

from video2aoiUtils import *

signatureImageDict= {}
yamlconfig = None

def p2ReadSignatureImage(k, value, c):
    '''Takes a key, a value (file name), and a context, and reads the image if key="match" or "track"
    then updates the global dict signatureImageDict'''
    global signatureImageDict, yamlconfig
    
    logging.getLogger('')

    if not isinstance(value, dict):
        # not a dict, no need to process
        return True
    
    # use the same YAML parsing method as p2Task()
    img = None
    if "match" in value:
        para=value["match"].split(",")
    elif "track" in value:
        para=value["track"].split(",")
    else:
        # not not something we want to process
        #logging.error("p2ReadSignatureImage: expecting match or track but got "+str(value))
        return True

    # set colorplane choices
    colorPlane = getColorPlane(yamlconfig)

    fname = para[0]
    if not isinstance(fname, str): 
        logging.error("p2ReadSignatureImage: expecting a filename but got "+str(fname))
        return True
    # if image path name is specified, can be absolute or relative
    if "imgFilePath" in yamlconfig["study"].keys():
        fname = os.path.join(yamlconfig["study"]["imgFilePath"], fname)

    # now read the image
    try:
        img = cv2.imread(fname)
        # check for optional sourceLoc parameters
        logging.info("p2ReadSignatureImage: reading image file="+str(fname))
    except:
        logging.error("p2ReadSignatureImage: error reading image file="+str(fname))
        return True

    if img is None:
        logging.error("p2ReadSignatureImage: error reading image file="+str(fname))
        print "p2ReadSignatureImage: error reading image file="+str(fname)
        exit(0)
       
    # extract the signature if sourceLoc is specified
    # this means that (a) fname is the src and (b) match will be attempted at this perceise location
    coord=[]
    if "sourceLoc" in value:
        # something like: sourceLoc: 836, 294, 256, 140
        coord = map(int, value["sourceLoc"].split(","))   # by default, in order x1, y1, x2, y2
        if "aoiFormat" in yamlconfig["study"]:
            if yamlconfig["study"]["aoiFormat"] == "xywh":
                # the x,y,w,h format: convert to xyxy format
                coord[2]=coord[2]+coord[0]+1
                coord[3]=coord[3]+coord[1]+1
        # now get the sig from the source image
        img= img[coord[1]:coord[3], coord[0]:coord[2]]

    destcoord = None
    if "destRange" in value:
        destcoord = map(int, value["destRange"].split(","))   # by default, in order x1, y1, x2, y2
        if "aoiFormat" in yamlconfig["study"]:
            if yamlconfig["study"]["aoiFormat"] == "xywh":
                # the x,y,w,h format: convert to xyxy format
                destcoord[2]=destcoord[2]+destcoord[0]
                destcoord[3]=destcoord[3]+destcoord[1]

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

    # now store the img into a dict, indexed by 
    # @@@ fname for now; it should be by the ITEM/AOI name, which shoud be unique, 
    #   because the fname is arbitrary and may be reused for different AOI definitions.
    # the dict will include the following info:
    #   - item/aoi name, i.e., the entry in YAML that has a child node "match"
    #   - fname, sourceLoc if necessary
    #   - colorPlane
    #   - destRange: this should be parsed only once, not per frame
    #   - lastKnownImg: None at the begining; set at the first match
    #   - template(s): a list of templates, specified 
    #   - lastKnownPosition: these will be set to None first but 
    sig={'id': "/".join(c + [k]),
        'fname':fname,   
        'img': img,
        'w': img.shape[1],
        'h': img.shape[0],
        'sourceLoc': coord,
        'destRange': destcoord,
        'colorPlane': colorPlane,
        'lastKnownPositionX': None,
        'lastKnownPositionY': None,
        'lastKnownBestFit': None,
        'currentFit': None
        }

    #signatureImageDict[fname]=img
    signatureImageDict[fname] = sig

    return True

def resizeSignatureImages(aoiScaleX, aoiScaleY, signatureImageDict):
    '''rescale images in the signatureImageDict by the xy scaling factors'''
    #global signatureImageDict
    for k in signatureImageDict:
        #signatureImageDict[k] = cv2.resize(signatureImageDict[k], (0,0), fx= aoiScaleX, fy=aoiScaleY) 
        sig = signatureImageDict[k]
        logging.debug("resizeSignatureImages: sig={}, size={}".format(k, np.shape(sig['img'])))

        sig['img'] = cv2.resize(sig['img'], (0,0), fx= aoiScaleX, fy=aoiScaleY) 
        sig['h']= sig['img'].shape[0]
        sig['w'] = sig['img'].shape[1]
        signatureImageDict[k] = sig

        logging.debug("resizeSignatureImages: ==> size={}".format(np.shape(sig['img'])))
        # for debugging only
        cv2.imwrite(sig['fname']+"_resized.png", sig["img"])
