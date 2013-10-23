import cv2
import cv2.cv as cv
import os
import os.path
import sys
import numpy as np
#import atexit
import glob
import logging
#import re
import subprocess

# import tesseract
# from HTMLParser import HTMLParser
# from htmlentitydefs import name2codepoint

# from csv import reader

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
        
# funcs to process the YAML config file
signatureImageDict={}
def p2ReadSignatureImage(k, fname, c):
    '''Takes a key, a value (file name), and a context, and reads the image if key="match" or "track"
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
    #logging.info("ColorPlane = "+str(colorPlane))

    
    # now get the last key, and if it's not "match" then return True to move to the next node
    if not k=="match" and not k=="track": 
        return True
    img = None
    # parse fname, because there may be optional parameters
    fname=fname.split(",")[0]

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
        sig=np.copy(signatureImageDict[fname])

        # optional threshold parameter
        threshold = -99        
        if len(para)==2:
            try:
                threshold = float(value["match"].split(",")[1])
            except:
                logging.error("MATCH: expecting a float number but got "+str(value["match"].split(",")[1]))
                return True
        # check for optional sourceLoc parameters
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
            sig= np.copy(signatureImageDict[fname][coord[1]:coord[3], coord[0]:coord[2]])

        # now let's find the template
        if threshold == -99:
            # use the global default threshold
            res = frameEngine.findMatch(frame, sig)
        else:
            # a new threshold is specified in the YAML file
            res = frameEngine.findMatch(frame, sig, threshold)
        
        if not res is None: 
            logging.debug("Lucky match using sourceLoc at "+str(coord))
        else:
            logging.debug("Didn't find using sourceLoc at "+str(coord))

        if res is None:
            # no match found; stop processing child nodes
            logging.debug("SignatureMatch: context="+str(context)+" fname="+str(fname)+" is not found in the current frame")
            return None
            # @@ unreachable code: not going to double check using slow mapping
            # if sourceLoc is specified, let's now search the whole image
            if "sourceLoc" in value:
                sig=np.copy(signatureImageDict[fname])
                # now let's find the template
                if threshold == -99:
                    # use the global default threshold
                    res = frameEngine.findMatch(frame, sig)
                else:
                    # a new threshold is specified in the YAML file
                    res = frameEngine.findMatch(frame, sig, threshold)
            if res is None:
                # if it's still None, or if it's still None,
                # no match found; stop processing child nodes
                logging.debug("SignatureMatch: context="+str(context)+" fname="+str(fname)+" is not found in the current frame")
                return None
        # if match fails, move to the next match; only proceed if Match succeeded

        # found match, whether it's the sourceLoc or the original image; 
        taskSigLoc, minVal=res
        logging.debug("MATCH: Signature="+str(fname)+"\tLocation="+str(taskSigLoc)+"\tminVal="+str(minVal)+txt)
        # @@ temporarily adding boxes for matching signatures. 
        h, w, clr= sig.shape
        updateAOI((str(fname), str(k), str(k), taskSigLoc[0], taskSigLoc[1], taskSigLoc[0]+w, taskSigLoc[1]+h))
            
    # track is like match in using templatematching, but it adds the object to the aoilist        
    if "track" in value:
        # first make sure v is in the signature image list
        #fname = value["track"]
        para=value["track"].split(",")
        fname = para[0]
        if not isinstance(fname, str): 
            logging.error("TRACK: expecting a filename but got "+str(fname))
            return True
        # if image path name is specified, can be absolute or relative
        if "imgFilePath" in yamlconfig["study"].keys():
            fname = os.path.join(yamlconfig["study"]["imgFilePath"], fname)
        if not (fname in signatureImageDict):
            logging.error("SignatureMatch: context="+str(context)+" fname="+str(fname)+" is not in the SignatureImageDict"+txt)
            return True
        # optional threshold parameter
        threshold = -99        
        if len(para)==2:
            try:
                threshold = float(value["track"].split(",")[1])
            except:
                logging.error("TRACK: expecting a float number but got "+str(value["track"].split(",")[1]))
                return True

        # now let's find the template
        if threshold == -99:
            # use the global default threshold
            res = frameEngine.findMatch(frame, signatureImageDict[fname])
        else:
            # a new threshold is specified in the YAML file
            res = frameEngine.findMatch(frame, signatureImageDict[fname], threshold)
            
        if res is None:
            # no match found; stop processing child nodes
            #logging.info("TRACK: context="+str(context)+" fname="+str(fname)+" is not found in the current frame")
            return None
        # if match fails, move to the next match; only proceed if Match succeeded
        else:
            # found match, break; print "==== Match! ==" + fname
            taskSigLoc, minVal=res
            logging.info("TRACK: Signature="+str(fname)+"\tLocation="+str(taskSigLoc)+"\tminVal="+str(minVal)+txt)
            h, w, clr= signatureImageDict[fname].shape
            updateAOI((str(fname), str(k), str(k), taskSigLoc[0], taskSigLoc[1], taskSigLoc[0]+w, taskSigLoc[1]+h))
            
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
    #logging.basicConfig(filename=logfilename, format='%(levelname)s\t%(asctime)s\t%(message)s', level=logging.DEBUG)
    logging.basicConfig(filename=logfilename, format='%(levelname)s\t%(relativeCreated)d\t%(message)s', level=logging.DEBUG)
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
        if lastCounter==frameNum-1: lastCounter = frameNum
        # @@ if moved forward, set skimmingMode = False
        #if lastCounter<frameNum-1:
        #    skimmingMode= False@@
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
            if processGazeLog and (gaze is not None and len(gaze) > 1 and len(aoilist)>1):
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
                # display rect for aoilist elements
                if "displayAOI" in yamlconfig["study"].keys() and yamlconfig["study"]["displayAOI"]==True:
                    if aoilist is not None:
                        for d in aoilist:
                            cv2.rectangle(frame, (d["x1"], d["y1"]), (d["x2"], d["y2"]) ,(255,0,0), 2)    
                            
                # shows the gaze circle
                if not np.isnan(gazex+gazey): 
                    cv2.circle(frame, (int(gazex), int(gazey)), 20, text_color)
                
                # displays the AOI of the last matched object
                if len(dump)>0: 
                    for d in dump:
                        cv2.rectangle(frame, (d["x1"], d["y1"]), (d["x2"], d["y2"]) ,text_color, 2)
                    cv2.rectangle(frame, (dump.x1[-1], dump.y1[-1]), (dump.x2[-1], dump.y2[-1]) ,text_color,2)
                # now show mouse, last pos; used to estimate toffset
                #curmouse = mouse[np.where(mouse.t<=vTime+ toffset)]
                curmouse = None
                if gaze is not None: curmouse = gaze[np.where(gaze.t<=vTime+ toffset)]
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
