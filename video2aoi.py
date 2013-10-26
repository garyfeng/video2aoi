import cv2
import cv2.cv as cv
import os
import os.path
import sys
import numpy as np
import glob
import logging
import subprocess

from TessEngine import *    #TessEngine, TessHTMLParser, imgTessHTMLParser
from FrameEngine import *    #FrameEngine

import yaml

    
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

def findLastMatchOffset():
    ''' This function looks into the aoilist for the last aoi with id="__MATCH__".
    If not found, it returns None. If found, it returns the coordinate of the upper left corner
    '''
    global aoilist

    offset=[-9999, -9999]
    contextLen = 0

    for d in aoilist:
        # look for the match with the longest context, which is the "deepest" match
        if d[1] == "__MATCH__" and len(d[0]) > contextLen:
            offset[0] = d[3]
            offset[1] = d[4]
            contextLen = len(d[0])
    # now done with searching, return None if nothing is found
    if offset[0] == -9999: return None
    # otherwise return the real deal
    return offset


# funcs to process the YAML config file
signatureImageDict={}
def p2ReadSignatureImage(k, value, c):
    '''Takes a key, a value (file name), and a context, and reads the image if key="match" or "track"
    then updates the global dict signatureImageDict'''
    global signatureImageDict
    
    if not isinstance(value, dict):
        # not a dict, no need to process
        return True
    
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
    #logging.info("ColorPlane = "+str(colorPlane))

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
        para=value["match"].split(",")
        fname = para[0]
        if not isinstance(fname, str): 
            logging.error("MATCH: expecting a filename but got "+str(fname))
            return True
        # if image path name is specified, can be absolute or relative
        if "imgFilePath" in yamlconfig["study"].keys():
            fname = os.path.join(yamlconfig["study"]["imgFilePath"], fname)
        if not (fname in signatureImageDict):
            logging.error("SignatureMatch: context="+str(context)+" fname="+str(fname)+" is not in the SignatureImageDict"+txt)
            return True

        # assuming the signature is the fname image, unless sourceLoc is specified later
        sig=signatureImageDict[fname]

        # optional threshold parameter
        threshold = -99        
        if len(para)==2:
            try:
                threshold = float(value["match"].split(",")[1])
            except:
                logging.error("MATCH: expecting a float number but got "+str(value["match"].split(",")[1]))
                return True
        # extract the signature if sourceLoc is specified
        # this means that (a) fname is the src and (b) match will be attempted at this perceise location
        img=np.copy(frame)
        srccoord=[0,0,0,0]
        if "sourceLoc" in value:
            # something like: sourceLoc: 836, 294, 256, 140
            srccoord = map(int, value["sourceLoc"].split(","))   # by default, in order x1, y1, x2, y2
            if "aoiFormat" in yamlconfig["study"]:
                if yamlconfig["study"]["aoiFormat"] == "xywh":
                    # the x,y,w,h format: convert to xyxy format
                    srccoord[2]=srccoord[2]+srccoord[0]
                    srccoord[3]=srccoord[3]+srccoord[1]
            # now get the sig from the source image
            #img= frame[srccoord[1]:srccoord[3], srccoord[0]:srccoord[2]]

        # now get the range of search in the destination
        # if not specified, we cut the img from sourceLoc (add 1 px in each deminsion so the alg works)
        # otherwise we cut the img out of frame based on destRange.
        destcoord = srccoord
        if not "destRange" in value:
            destcoord[2]=destcoord[2]+1
            destcoord[3]=destcoord[3]+1
        else:
            # destRange is in the value
            destcoord = map(int, value["destRange"].split(","))   # by default, in order x1, y1, x2, y2
            if "aoiFormat" in yamlconfig["study"]:
                if yamlconfig["study"]["aoiFormat"] == "xywh":
                    # the x,y,w,h format: convert to xyxy format
                    destcoord[2]=destcoord[2]+destcoord[0]
                    destcoord[3]=destcoord[3]+destcoord[1]

        # we now have the dest range; now use this to cut the image
        img= frame[destcoord[1]:destcoord[3], destcoord[0]:destcoord[2]]

        # now let's find the template
        if threshold == -99:
            # use the global default threshold
            res = frameEngine.findMatch(img, sig)
        else:
            # a new threshold is specified in the YAML file
            res = frameEngine.findMatch(img, sig, threshold)
        
        if res is None:
            # no match found; stop processing child nodes
            logging.debug("MATCH: context="+str(context)+" fname="+str(fname)+" is not found in the current frame")
            if "unmatchLog" in value:
                # need to log this event
                logging.info("LOG\t"+txt+"\tcontext='"+str(context)+"'\tmsg='"+value["unmatchLog"]+"'")
            return None
        # only proceed if Match succeeded
        taskSigLoc, minVal=res
        objoffset = [taskSigLoc[0] + destcoord[0], taskSigLoc[1] + destcoord[1]]

        coord=[0,0,0,0]
        h, w, clr= sig.shape
        coord[0]= objoffset[0]
        coord[1]= objoffset[1]
        coord[2]= w+ objoffset[0]
        coord[3]= h+ objoffset[1]

        logging.debug("MATCH:\t"+txt+"\tSignature="+str(fname)+"\tLocation="+str(objoffset)+" AOI="+str(coord)+"\tminVal="+str(minVal))
        updateAOI((str(fname), "__MATCH__", str(k), coord[0], coord[1], coord[2], coord[3]))

        # # found match, whether it's the sourceLoc or the original image; 
        # taskSigLoc, minVal=res
        # logging.debug("MATCH: Signature="+str(fname)+"\tLocation="+str(taskSigLoc)+"\tminVal="+str(minVal)+txt)
        # # @@ temporarily adding boxes for matching signatures. 
        # h, w, clr= sig.shape
        # #@@ need to use coord()
        # if "sourceLoc" in value:
        #     updateAOI((str(fname), "__MATCH__", str(k), coord[0], coord[1], coord[2], coord[3]))
        # else:
        #     updateAOI((str(fname), "__MATCH__", str(k), taskSigLoc[0], taskSigLoc[1], taskSigLoc[0]+w, taskSigLoc[1]+h))


    # if successful match or NO match needed
    if "log" in value: 
        # simply log the message
        logging.info("LOG\t"+txt+"\tcontext='"+str(context)+"'\tmsg='"+value["log"]+"'")
    if "aoi" in value:
        # an AOI defined directly by coordinates; will output and add the aoi for matching
        coord = map(int, value["aoi"].split(","))   # by default, in order x1, y1, x2, y2
        if "aoiFormat" in yamlconfig["study"]:
            if yamlconfig["study"]["aoiFormat"] == "xywh":
                # the x,y,w,h format: convert to xyxy format
                coord[2]=coord[2]+coord[0]
                coord[3]=coord[3]+coord[1]
        pageTitle = "/".join(context)        # 'Assessment/items/Task3DearEditor/tab1', only path to the parent 
        logging.info("AOIDAT\t"+txt+"\t"+pageTitle+"\t"+str(k)+"\t"+'\t'.join(map(str, coord))+"\t"+str(k))
        updateAOI((pageTitle, str(k), str(k), coord[0], coord[1], coord[2], coord[3]))

    if "relativeAOI" in value:
        # something like: relativeAOI: 0, 0, 785, 573
        # 
        # first, find the latest __MATCH__ in the aoilist, and return the offset
        objoffset = findLastMatchOffset()
        if objoffset is None:
            # error, most likely because there is no __MATCH__ in aoilist
            logging.error("relativeAOI: Cannot find the last matched object. No AOI output")
            return None

        # read in the relative para
        coord = map(int, value["relativeAOI"].split(","))   # by default, in order x1, y1, x2, y2
        if "aoiFormat" in yamlconfig["study"]:
            if yamlconfig["study"]["aoiFormat"] == "xywh":
                # the x,y,w,h format: convert to xyxy format
                coord[2]=coord[2]+coord[0]
                coord[3]=coord[3]+coord[1]
        coord[0]=coord[0]+ objoffset[0]
        coord[1]=coord[1]+ objoffset[1]
        coord[2]=coord[2]+ objoffset[0]
        coord[3]=coord[3]+ objoffset[1]
        # output
        pageTitle = "/".join(context)
        logging.info("AOIDAT\t"+txt+"\t"+pageTitle+"\t"+str(k)+"\t"+'\t'.join(map(str, coord))+"\t"+str(k))
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
        logging.info("OCR\t"+txt+"\tcontext='"+str(context)+"'\tcoord='"+str(coord)+"'\tconfidence="+str(tess.confidence)+"\ttext='"+ ocrtext[:15]+"'")
        html=tess.getHtml() # log the HTML values
        # logging OCR results?
        if "ocrLogText" in yamlconfig["study"] and yamlconfig["study"]["ocrLogText"]:
            logging.info("OCRTEXTBGIN\t"+ocrtext+"\tOCRTEXTEND:\n")
        if "ocrLogHTML" in yamlconfig["study"] and yamlconfig["study"]["ocrLogHTML"]:
            logging.info("\nOCRAOIBEGIN:\n"+html+"\nOCRAOIEND:\n")
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
        logging.info(txt+"\tScreenshot f='"+fname+"'"+txt)
    # Dealing with special commands: break, to not continue processing the rest of the list
    # if "break" in value:
    if "break" in k:
        # skip the rest of the tests in the same level
        #if  value["break"]:
        #    logging.info("Breaking...")
        #    return False
        logging.debug("Breaking...")
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
    logLevel = logging.INFO
    if "logLevelDebug" in yamlconfig["study"] and yamlconfig["study"]["logLevelDebug"]:
        logLevel= logging.DEBUG
    #logging.basicConfig(filename=logfilename, format='%(levelname)s\t%(asctime)s\t%(message)s', level=logging.DEBUG)
    logging.basicConfig(filename=logfilename, format='%(levelname)s\t%(relativeCreated)d\t%(message)s', level=logLevel)
    
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
    # moving to the startFrame
    video.set(cv.CV_CAP_PROP_POS_FRAMES, startFrame)

    # log
    logging.info("video = "+str(v)+"\tScaling ratio =" +str(ratio) +"\tlog = '"+str(logfilename)+"'")
    logging.info("VideoFrameSize = "+str(video.get(cv.CV_CAP_PROP_FRAME_WIDTH ))+"\t"+str(video.get(cv.CV_CAP_PROP_FRAME_HEIGHT )))
    logging.info("NumFrames = "+str(nFrames)+"\tStartFrame = "+str(startFrame)+ "\tFPS="+str(fps))

    # read signature image files for template matching
    p2YAML(yamlconfig["tasks"], p2ReadSignatureImage)
    
    # read eye event logs, only if doNotProcessGazeLog=False or unspecified
    basename = os.path.basename(os.path.splitext(v)[0])
    processGazeLog = True
    gaze=None; mouse=None;

    if "processGazeLog" in yamlconfig["study"].keys():
        processGazeLog= yamlconfig["study"]["processGazeLog"]
    logging.info("processGazeLog = "+str(processGazeLog))
    
    if processGazeLog:
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
    aoilist=[]; dump=[]; lastCounter=0; lastVTime=0;
    
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

    ###############################
    # now loop through the frames
    ###############################
    while video.grab():
        frameNum = video.get(cv.CV_CAP_PROP_POS_FRAMES)
        videoHeadMoved = False    # this is used to keep track of video seeking

        # lastCounter tracks the greedy jumpahead position, which should be within skimFrames
        # when in skimmingMode, both of these should advance; this includes when the user jumps ahead with the slider
        logging.debug("V: frameNum="+str(frameNum)+"\tlastCounter="+str(lastCounter)+"\tskimmingMode= "+str(skimmingMode))
        
        # normal case, reading one frame at once
        if lastCounter==frameNum-1: lastCounter = frameNum
        # if in the refined search mode and the frameNum catches with lastCounter, then we resume skimming
        if not skimmingMode and frameNum == lastCounter :
            # not in skimmingMode but we have scanned all the frames in between
            skimmingMode = True

        # jumpping forward multiple frames (e.g., using the slidebar); we stop the skimmingMode
        # if lastCounter is way ahead of frameNum, it's clear that it's caused by user rewind; reset
        if lastCounter<frameNum-1 or lastCounter >frameNum + skimFrames+1: 
            logging.info("Video jumpping: frameNum="+str(frameNum)+"\tlastCounter="+str(lastCounter)+"\tskimmingMode= "+str(skimmingMode))
            lastCounter = frameNum
            skimmingMode = False
            videoHeadMoved = True
            # here you don't want to clear the frameEngine.lastFrame, because you want to detect the first change.

        # skimming mode: skipping frames at a time
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
            if videoHeadMoved: 
                lastVTime = vTime     # otherwise whenever there is a jump we will export all the gaze in between.

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
            # logging.debug("SkimmingMode ="+str(skimmingMode)+", lastCounter= "+str(lastCounter)+" frameNum= "+str(frameNum)+" skimFrames= "+str(skimFrames))
            if (frameChanged and skimmingMode and frameNum>skimFrames):
                # now we need to rewind and redo this in the normal mode
                skimmingMode = False
                lastCounter = frameNum+1   #lastCounter tracks where we have skimmed to, but add 1, else in rare cases it's thrown into a loop
                frameNum = frameNum-skimFrames-1    # going back 1 more frame than before because we clearLastFrame() and need this to establish the baseline
                video.set(cv.CV_CAP_PROP_POS_FRAMES, frameNum)
                logging.debug("SkimmingMode: Changed detected; going back\tskimmingMode="+str(skimmingMode)+"\tlastCounter="+str(lastCounter)+"\tframeNum="+str(frameNum)+"\tskimFrames= "+str(skimFrames))
                # going back, rewind to frameNum
                frameEngine.clearLastFrame()    # clear the lastFrame buffer, so that the first rewinded frame will be taken as the template to compare with.
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
            if processGazeLog and (gaze is not None and len(gaze) > 1):
                # @@ this is where we should recalibrate 
                # (a) redefine lineHeight and wordWidth so that no gap is left
                # (b) get heatmap and estimate distribution to the left and top
                  # edges and other "good features"; calc teh best fit
                # (c) does kernalDensity or bleeding, so that we get a matrix
                  # of the "activation" on each AOI over time

                # the original algorithm only gets the last gaze sample 
                # we need to report on all gaze samples that fall between this and last video frame that has been processed, tracked by lastVTime
                # see http://stackoverflow.com/questions/12647471/the-truth-value-of-an-array-with-more-than-one-element-is-ambigous-when-trying-t
                temp = gaze[np.where(np.logical_and(gaze.t>lastVTime+toffset, gaze.t<=vTime+toffset))]   
                #print str(lastVTime) +"-"+str(vTime) +"="+ str(vTime-lastVTime)

                lastVTime = vTime   # used to track gazes during skimming.
                for g in temp:
                    gazetime= g["t"]
                    gazex=int(g["x"])
                    gazey=int(g["y"])
                    gazeinfo= g["info"]

                    # vTime is the time of the current video frame, which, in the case of skimming, may have skipped several frames from the last check.
                    # if we use vTime in the output, we can't tell the exact video time
                    # so we back calculate here from gt:
                    videoTime = gazetime -toffset

                    # now need to find the AOI and log it
                    # this means that the p2Task() need to have a global AOI array
                    if  len(aoilist)<1:
                        # a page without AOI, mostly likely junk 
                        #logging.info("Gaze\tvt="+str(vTime)+"\tgzt="+str(gazetime)+"\tx="+str(gazex)+"\ty="+str(gazey)+"\tinfo="+str(gazeinfo)+"\taoi=JUNKSCREEN"+"\t\t\t\t\t\t")
                        logging.info("Gaze\tvt="+str(videoTime)+"\tgzt="+str(gazetime)+"\tx="+str(gazex)+"\ty="+str(gazey)+"\tinfo="+str(gazeinfo)+"\taoi=JUNKSCREEN"+"\t\t\t\t\t\t")
                    elif not np.isnan(gazex+gazey)  and gazetime !=lastGazetime:
                        # aoilist is defined
                        dump=aoilist[np.where(aoilist.x1<=gazex )]
                        dump=dump[np.where(dump.x2>gazex)]
                        dump=dump[np.where(dump.y1<=gazey)]
                        dump=dump[np.where(dump.y2>gazey)]
                        if len(dump)>0:
                            for aoi in dump:
                                if not "__MATCH__" in aoi["id"]:
                                    # skip templates for matching or tracking
                                    logging.info("Gaze\tvt="+str(videoTime)+"\tgzt="+str(gazetime)+"\tx="+str(gazex)+"\ty="+str(gazey)+"\tinfo="+str(gazeinfo)+"\taoi="+"\t".join([str(s) for s in aoi]))
                        else:
                            # gaze is not on an AOI; print out the name of the page; keep in mind that aoilist[0] is the match template 
                            logging.info("Gaze\tvt="+str(videoTime)+"\tgzt="+str(gazetime)+"\tx="+str(gazex)+"\ty="+str(gazey)+"\tinfo="+str(gazeinfo)+"\taoi="+str(aoilist[-1]["page"])+"\t\t\t\t\t\t")
                    else:
                        # invalid gazex or gazey
                        logging.info("Gaze\tvt="+str(videoTime)+"\tgzt="+str(gazetime)+"\tx=-9999"+"\ty=-9999"+"\tinfo="+str(gazeinfo)+"\taoi="+str(aoilist[0]["page"])+"\t\t\t\t\t\t")

                    # tracking things here
                    lastGazetime=gazetime     

            # end of AOI
            ############################
            # display video
            ############################
            if showVideo:
                text_color = (128,128,128)
                txt = txt+"\t"+str(parser.ocrPageTitle)
                cv2.putText(frame, txt, (20,100), cv2.FONT_HERSHEY_PLAIN, 1.0, text_color, thickness=1)
                # display rect for aoilist elements
                if "displayAOI" in yamlconfig["study"].keys() and yamlconfig["study"]["displayAOI"]==True:
                    if aoilist is not None:
                        for d in aoilist:
                            if "__MATCH__" in d["id"]:
                                # matching or tracking images
                                cv2.rectangle(frame, (d["x1"], d["y1"]), (d["x2"], d["y2"]), (0,255,0), 2)
                            else:
                                # actual AOIs
                                cv2.rectangle(frame, (d["x1"], d["y1"]), (d["x2"], d["y2"]) ,(255,0,0), 2)    
                            
                # shows the gaze circle
                if not np.isnan(gazex+gazey): 
                    cv2.circle(frame, (int(gazex), int(gazey)), 20, text_color)
                
                # displays the AOI of the last matched object
                if not aoilist is  None and len(dump)>0: 
                    for d in dump:
                        if not "__MATCH__" in d["id"]:
                            # actual active AOIs
                            cv2.rectangle(frame, (d["x1"], d["y1"]), (d["x2"], d["y2"]), (0,0,255), 2)

                    #cv2.rectangle(frame, (dump.x1[-1], dump.y1[-1]), (dump.x2[-1], dump.y2[-1]), text_color,2)
                # now show mouse, last pos; used to estimate toffset
                #curmouse = mouse[np.where(mouse.t<=vTime+ toffset)]
                curmouse = None
                if gaze is not None: 
                    curmouse = gaze[np.where(gaze.t<=vTime+ toffset)]
                    if curmouse is not None and len(curmouse)>0: 
                        cv2.circle(frame, (int(curmouse.x[-1]), int(curmouse.y[-1])), 10, (0,0,255), -1)
                    
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
        print "Usage: python video2aoi.py config.yaml videofile.avi [-novideo]"
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
    frameEngine.matchTemplateThreshold=float(yamlconfig["study"]["matchTemplateThreshold"])


    # init global vars
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
