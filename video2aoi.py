import cv2
import cv2.cv as cv
import os
import os.path
#import sys
import numpy as np
import glob
import logging
import subprocess
import argparse


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
        
    # here we enforce that the item name is in the "page" field
    # data[0]=data[0].replace("Assessment/items", "")
    # if len(data[0])==0:
    #     # if it's Assessment/items, then the aoiID becomes the page name
    #     data[0]=data[1]

    aoilist.append(data)

def findVideoGazeOffset(mouseLog, videoMouseLocations, locationThreshold = 2, temporalThreshold = 17):
    '''Given the mouseLog (in numpy array) and videoMouseLocations (a numpy array) that contains
    some fractions of mouse locations found in the video frames via template matching, find the 
    most likely time shift parameter. None if no offset parameter can be found. 

    locationThreshold is the parameter for difference in x,y locations between the mouseLog and 
    videoMouseLocations; default is 2 pixels in either x or y.

    temporalThreshold is the parameter for the search window. 

    '''
    if not isinstance(mouseLog, np.ndarray):
        print "Error findVideoGazeOffset(): mouseLog is not a numpy array"

    pass

def findLastMatchOffset(context):
    ''' This function parses the 'context' tree, looks into the aoilist for the last aoi with id="__MATCH__".
    If not found, it returns None. If found, it returns the coordinate of the upper left corner.

    context is a list. 
    '''
    global aoilist

    if context is None: 
        logging.debug("findLastMatchOffset: input context = '{}' is None".format(str(context)))
        return None
    if not isinstance(context, list): 
        logging.debug("findLastMatchOffset: input context = '{}' is not a list".format(str(context)))
        return None

    # reverse context; must make a copy, or else it messes up the context
    c = context[:]  # can't do c=context, which simply makes a reference. 
    c.reverse()

    offset=[-9999, -9999]
    #contextLen = 0

    for d in aoilist:
        # look for the match with the longest context, which is the "deepest" match
        # if d[1] == "__MATCH__" and len(d[0]) > contextLen:
        #     offset[0] = d[3]
        #     offset[1] = d[4]
        #     contextLen = len(d[0])
        # look for the match with the longest context, which is the "deepest" match
        #print "findLastMatchOffset: input context = '{}' ".format(str(context))
        for key in c:
            if d[1] == "__MATCH__"+str(key):
                offset[0] = d[3]
                offset[1] = d[4]
                break;
    # now done with searching, return None if nothing is found
    #if offset[0] == -9999: return None
    # otherwise return the real deal
    return None if offset[0] == -9999 else offset

def getColorPlane():
    '''Returns the color plane code specified in the YAML file'''
    global yamlconfig

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
    return colorPlane    

def readEventData(basename):
    ''' read the event log file as specified in basename and suffix in the yaml file.
    Returns the data as a numpy Recarray. Else return None.
    '''
    global yamlconfig
    # get the gaze/key/mouse data file name
    if "dataFileSuffix" in yamlconfig["study"].keys():
        datafilename = basename + yamlconfig["study"]["dataFileSuffix"]
    else:
        datafilename = basename + "_events.txt"; #default
    print "processGazeLog: datafilename="+datafilename

    # check to see if it's empty; if so, delete it
    if os.path.isfile(datafilename) and os.path.getsize(datafilename)==0:
        print('processGazeLog: Eyelog2Dat: %s file is empty. Deleted' % datafilename)
        os.remove(datafilename)
    if not os.access(datafilename, os.R_OK):
        # getting ready the process the *eye_log file to generate the datafile
        if "datalogFileSuffix" in yamlconfig["study"].keys():
            datalogfilename = basename + yamlconfig["study"]["datalogFileSuffix"]
        else:
            datalogfilename = basename + "_eye.log"; #default
        print('processGazeLog: Eyelog2Dat: %s file is not present' % datafilename)
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
            print("processGazeLog: Error calling "+awkfile)
            logging.error("processGazeLog: Error calling " +awkfile)
            return None

    # read all data
    try:
        alldata = np.genfromtxt(datafilename, delimiter='\t', dtype=None, names=['t', 'event', 'x', 'y', 'info'])
    except:
        # no gaze file to read; fake one
        logging.error("processGazeLog: file "+datafilename+" cannot be read.")
        return None
        #alldata = np.genfromtxt("fake_events.txt", delimiter='\t', dtype=None, names=['t', 'event', 'x', 'y', 'info'])
    
    # process all the data, separate gaze/key/mouse events    
    alldata = alldata.view(np.recarray)    # now you can refer to gaze.x[100]
    return alldata

def logEvents (allevents, aoilist, lastVTime, vTime, tOffset=0):
    '''Function to extract event info, and a time window, and log events inside the window.
    Returns True if all goes well, or False if something is wrong.
    allevents is of the format names=['t', 'event', 'x', 'y', 'info']
    aoilist is of the format  dtype=[('page', 'S80'), ('id', 'S40'), ('content','S80'), ('x1',int), ('y1',int), ('x2',int), ('y2',int)]
    '''

    # not the best idea but we need to keep track of these for displaying the gaze data. 
    # unless we want to re-calculate these every time
    global gazex, gazey, mousex, mousey, activeAOI

    if len(allevents)==0: 
        logging.error("logEvents: no event data")
        return False
    if (lastVTime>vTime):
        logging.error("logEvents: lastVTime {} > vTime {}. Flipping them".format(lastVTime, vTime))
        dump = vTime
        vTime=lastVTime
        lastVTime=dump

    # set gaze pos to missing, but not mouse pos
    gazex=-32768; gazey=-32768;

    # debug: output AOIs
    for a in aoilist:
        logging.debug("logEvents: AOI = {}".format("\t".join([str(s) for s in a])))
 
    # the original algorithm only gets the last gaze sample 
    # we need to report on all gaze samples that fall between this and last video frame that has been processed, tracked by lastVTime
    # see http://stackoverflow.com/questions/12647471/the-truth-value-of-an-array-with-more-than-one-element-is-ambigous-when-trying-t
    #temp = gaze[np.where(np.logical_and(gaze.t>lastVTime+toffset, gaze.t<=vTime+toffset))]   
    frameEvents = allevents[np.where(np.logical_and(allevents.t>lastVTime+toffset, allevents.t<=vTime+toffset))]  
    # sort by time so that the output is in order 
    frameEvents.sort(order="t")
    for e in frameEvents:
        etime = int(e["t"])
        # vTime is the time of the current video frame, which, in the case of skimming, may have skipped several frames from the last check.
        # if we use vTime in the output, we can't tell the exact video time
        # so we back calculate here from gt:
        videoTime = etime -toffset
        estring = "{}:\tvt={}\tgzt={}\tx={}\ty={}\tinfo={}".format(e["event"], int(videoTime), etime, e["x"], e["y"], e["info"])

        try:        
            x=int(e["x"])
            y=int(e["y"])
        except:
            # not available
            x=-32768; y=-32768

        # you shouldn't have a case whereaoistring is undefined without the follow ling but it had occurred. 
        currItem = "NONAOI"
        aoistring = currItem+"\t\t\t\t\t\t"
        for a in aoilist:
            #logging.debug("logEvents: aoi page = {}".format(a["page"]))
            # if the page title starts with Assessment/items, then this is the page, and we exit this loop
            if a["page"].startswith("Assessment/items"):
                currItem = a["page"]
                logging.debug("logEvents: currItem = {}".format(currItem))
                break

        # Now start to assign AOI, for each matching AOI; if there are no matching AOIs, we print the page title        
        if len(aoilist)==0:
            aoistring = "JUNKSCREEN"+"\t\t\t\t\t\t"
            activeAOI=[]  
        elif x>0 and y>0:
            # this skips keystrokes and missing data, junk screen etc.
            # now do aoi matching
            activeAOI=aoilist[np.where(aoilist.x1<=x )]
            activeAOI=activeAOI[np.where(activeAOI.x2>x)]
            activeAOI=activeAOI[np.where(activeAOI.y1<=y)]
            activeAOI=activeAOI[np.where(activeAOI.y2>y)]
            if len(activeAOI)>0:
                for aoi in activeAOI:
                    #if not "__MATCH__" in aoi["id"]:
                    if not aoi["id"].startswith("__MATCH__"):
                        # skip templates for template matching or tracking
                        aoistring="\t".join([str(s) for s in aoi])
            else:
                # gaze is not on an AOI; print out the name of the page; 
                # keep in mind that aoilist[0] is the match template, which is not useful; 
                # [-1] is often some other junk model thrown at the Assessment level, which we often don't want
                # this should have a persistent "item" code even if the gaze is not on the item 
                aoistring = str(currItem)+"\t\t\t\t\t\t"
        else:
            # for keystrokes or bad gaze data, etc. at least print the page
            aoistring = str(currItem)+"\t\t\t\t\t\t"

        # now let's log
        logging.info(estring +"\taoi=" +aoistring)

        # now track the last gaze and the last mouse events for this "frame"
        #@@ this is ugly -- using global vars
        # should be something like
        #  frameEngine.updateCurrentGaze(x,y)
        if "gaze" in e["event"]:
            # update gaze pos no matter what; missing data -> missing
            gazex =x; gazey =y; 
        elif "mouse" in e["event"] and x>0 and y>0:
            # don't update mosue location if there is no mouse info in this frame
            mousex=x; mousey=y;

    return True

def displayFrame(windowName):
    '''Shows the current frame of video, along with AOI and gaze/mouse '''
    global frame, txt, yamlconfig, aoilist, vTime
    global gazex, gazey, mousex, mousey, activeAOI

    text_color = (128,128,128)
    txt+= ", gaze=({}, {}), mouse=({}, {})".format(gazex, gazey, mousex, mousey)
    cv2.putText(frame, txt, (20,100), cv2.FONT_HERSHEY_PLAIN, 1.0, text_color, thickness=1)
    # display rect for aoilist elements
    if "displayAOI" in yamlconfig["study"].keys() and yamlconfig["study"]["displayAOI"]==True:
        # if aoilist is not None:
        for d in aoilist:
            #if "__MATCH__" in d["id"]:
            if d["id"].startswith("__MATCH__"):
                # matching or tracking images
                cv2.rectangle(frame, (d["x1"], d["y1"]), (d["x2"], d["y2"]), (0,255,0), 2)
            else:
                # actual AOIs
                cv2.rectangle(frame, (d["x1"], d["y1"]), (d["x2"], d["y2"]) ,(255,0,0), 2)    
    
    # displays the AOI of the last matched object
    if len(aoilist)>0 and len(activeAOI)>0: 
        for d in activeAOI:
            #if not ("__MATCH__" in d["id"]):
            if not d["id"].startswith("__MATCH__"):
                # actual active AOIs
                cv2.rectangle(frame, (d["x1"], d["y1"]), (d["x2"], d["y2"]), (0,0,255), 2)
                
    # shows the gaze circle
    if not np.isnan(gazex+gazey): 
        cv2.circle(frame, (int(gazex), int(gazey)), 10, (0,0,255), -1)

    # now show mouse, last pos; used to estimate toffset
    if not np.isnan(mousex+mousey):
        cv2.circle(frame, (int(mousex), int(mousey)), 20, (0,0,255), 2)
        
    cv2.imshow(windowName, frame)       # main window with video control
        
    # keyboard control; ESC = quit
    key= cv2.waitKey(1) #key= cv2.waitKey(waitPerFrameInMillisec)
    if (key==27):
        logging.info("GUI: ESC pressed"+txt)
        return False
    elif (key==32):
        logging.info("GUI: paused"+txt)
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
        logging.info("GUI: key="+str(key)+"\tvideoFrame written to="+windowName+"_"+str(vTime)+".png"+txt)
        cv2.imwrite(windowName+"_"+str(vTime)+".png", frame)
        print key
    else: pass

    return True

# funcs to process the YAML config file
signatureImageDict={}
def p2ReadSignatureImage(k, value, c):
    '''Takes a key, a value (file name), and a context, and reads the image if key="match" or "track"
    then updates the global dict signatureImageDict'''
    global signatureImageDict, yamlconfig
    
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
    colorPlane = getColorPlane()

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
    signatureImageDict[fname]=img
    return True
                
def p2Task(k, value, context):
    '''A callback function for p2YAML(). It taks a list of keys (context) and a Value from the
    yamlconfig, and takes appropriate actions; 
    returns 
        None if no-match, ==> stop processing any subnodes
        True if we want to continue processing the rest of the sibling elements ; 
        False to stop processing sibling elements

        Note how it works: It only processes "dict" types -- i.e., AOI definition entries. 
        It won't process any other types of YAML entries. 
        With an dict entry, it looks INSIDE the entry to see process subnodes explicitly.   
        If the subnode is a dict type, it gets processed iteratively.  But things like "AOI"
        or "OCR" won't get processed unless it's under a dict entry.  
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
        if "sourceLoc" in value:
            destcoord = srccoord
        else:
            # no sourceLoc specified, we use the whole frame; 
            # note the shape() func returns [h,w,#color] as in numpy
            destcoord = [0,0, frame.shape[1], frame.shape[0]]

        if not "destRange" in value:
            # let's do +/- 2 pix on each side
            destcoord[0]=destcoord[0]-2
            destcoord[1]=destcoord[1]-2
            destcoord[2]=destcoord[2]+2
            destcoord[3]=destcoord[3]+2
        else:
            # destRange is in the value
            destcoord = map(int, value["destRange"].split(","))   # by default, in order x1, y1, x2, y2
            if "aoiFormat" in yamlconfig["study"]:
                if yamlconfig["study"]["aoiFormat"] == "xywh":
                    # the x,y,w,h format: convert to xyxy format
                    destcoord[2]=destcoord[2]+destcoord[0]
                    destcoord[3]=destcoord[3]+destcoord[1]

        # make sure everything is within the frame
        if destcoord[0]<0: destcoord[0]=0
        if destcoord[1]<0: destcoord[1]=0
        if destcoord[2]>frame.shape[1]: destcoord[2]=frame.shape[1]
        if destcoord[3]>frame.shape[0]: destcoord[3]=frame.shape[0]

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
        updateAOI((str(fname), "__MATCH__"+str(k), str(k), coord[0], coord[1], coord[2], coord[3]))


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
        pageTitle = "/".join(context)        # 'Assessment/items/Task3DearEditor'
        logging.info("AOIDAT\t"+txt+"\t"+pageTitle+"\t"+str(k)+"\t"+'\t'.join(map(str, coord))+"\t"+str(k))
        updateAOI((pageTitle, str(k), str(k), coord[0], coord[1], coord[2], coord[3]))

    if "relativeAOI" in value:
        # something like: relativeAOI: 0, 0, 785, 573
        # 
        # first, find the latest __MATCH__ in the aoilist, and return the offset
        # sending the context string, so that the function can parse and find the last match object
        objoffset = findLastMatchOffset(context+[k])
        if objoffset is None:
            # error, most likely because there is no __MATCH__ in aoilist
            logging.error("relativeAOI: Cannot find the last matched object '{}'. No AOI output".format(k))
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
        #print fname
        try:
            cv2.imwrite(fname, frame)
        except:
            logging.error("Error writing the current frame as a screenshot to file ="+str(fname))
        logging.info(txt+"\tScreenshot f='"+fname+"'"+txt)
    # Dealing with special commands: break, to not continue processing the rest of the list
    if "break" in value:
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

    global yamlconfig, gaze, aoilist, toffset, logLevel, outputLogFileSuffix
    global video, frame,  startFrame #, minVal, taskSigLoc,
    global txt, vTime, jumpAhead, skimmingMode
    global gazex, gazey, mousex, mousey, activeAOI
    
    # local vars
    alldata=None

    # init vars
    try:
        ratio = yamlconfig["study"]["scalingRatio"]
    except:
        ratio = 1
    
    # create new log for the file v
    basename = os.path.basename(os.path.splitext(v)[0])

    try:
        logfilename = basename+outputLogFileSuffix
    except:
        logfilename = basename+"_AOI.log"

    logfilename = os.path.join(os.getcwd(), logfilename)

    logging.basicConfig(filename=logfilename, format='%(levelname)s\t%(relativeCreated)d\t%(message)s', level=logLevel)
    # nothing will be written for testMode. It will create a new file with zero byte, though. 
    if testMode:
        print("program running in testMode; no data saved to {}".format(logfilename))
        logging.disable(logging.INFO)

    if not testMode: 
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

    # read signature image files for template matching
    p2YAML(yamlconfig["tasks"], p2ReadSignatureImage)
    
    # read eye event logs, only if doNotProcessGazeLog=False or unspecified
    processGazeLog = True

    gaze=None; 

    if "processGazeLog" in yamlconfig["study"].keys():
        processGazeLog= yamlconfig["study"]["processGazeLog"]
    logging.info("processGazeLog: processGazeLog = "+str(processGazeLog))
    
    if processGazeLog:

        # read in alldata
        alldata = readEventData(basename)
        if  alldata is None:
            # no data read
            logging.error("processGazeLog: error reading eye gaze data for {}. Skipping this file".format(basename))
            print "processGazeLog: error reading eye gaze data for {}. Skipping this file".format(basename)
            return False

        gaze = alldata[np.where(alldata.event=="gaze")]
        print "processGazeLog: gaze data len = "+str(len(gaze))
        
        #gaze = [row for row in reader(datafilename, delimiter='\t') if row[1] == 'gaze']
        if(gaze is not None and len(gaze) < 1):
            print("Error reading gaze data! File="+basename)
            logging.error("Error reading gaze data! File="+basename)
        print "Gaze read from "+basename +" with n="+str(len(gaze))
        # end reading eye event log
    
    # now let's skip the video to the first gaze time, but only if startFrame is not deberately set.
    if startFrame <=1 and gaze is not None:
        # translate from gaze time to vTime
        startFrame = int((gaze.t[0] - toffset) * fps /1000) -fps    # less a second 
        if startFrame<1: startFrame=1
        print "startFrame ="+ str(startFrame)+" gaze.t[0]="+str(gaze.t[0]) + " toffset="+str(toffset) + " fps=" +str(fps)

    # moving to the startFrame
    video.set(cv.CV_CAP_PROP_POS_FRAMES, startFrame)
    logging.info("NumFrames = "+str(nFrames)+"\tStartFrame = "+str(startFrame)+ "\tFPS="+str(fps))

    # init
    gazex=-999; gazey=-999; mousex=-99; mousey=-99; activeAOI=[]

    # set the flag for skimmingMode
    skimmingMode=True; frameChanged=False; skimFrames = int(jumpAhead * fps)
    aoilist=[]; lastCounter=0; lastVTime=0;

    # set colorplane choices
    colorPlane = getColorPlane()

    ###############################
    # now loop through the frames
    ###############################
    while video.grab():
        frameNum = video.get(cv.CV_CAP_PROP_POS_FRAMES)
        videoHeadMoved = False    # this is used to keep track of video seeking

        # lastCounter tracks the greedy jumpahead position, which should be within skimFrames
        # when in skimmingMode, both of these should advance; this includes when the user jumps ahead with the slider
        logging.debug("V: frameNum="+str(frameNum)+"\tlastCounter="+str(lastCounter)+"\tskimmingMode= "+str(skimmingMode)+"\tskimFrames= "+str(skimFrames))
        
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
        if skimmingMode and skimFrames>0 and  frameNum % skimFrames >0 : continue
        
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

        ###############################
        # if no error reading the frame  
        ###############################  
        if flag:    
            # captions
            vTime = video.get(cv.CV_CAP_PROP_POS_MSEC)
            if videoHeadMoved: 
                lastVTime = vTime     # otherwise whenever there is a jump we will export all the gaze in between.

            txt="\tvideo='"+v+"'\tt="+str(vTime) +'\tframe='+str(frameNum)

            ################################################
            # now only process when there is a large change
            ################################################
            # in the case no jumpAhead is set, always assume the frame is changed.
            if jumpAhead >0:
                frameChanged = frameEngine.frameChanged(frame) 
            else:
                # no jummping ahead; so process every frame without change-detection.
                frameChanged = True
                skimmingMode = False

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
                # now go through the tasks and items
                aoilist = []
                p2YAML(yamlconfig["tasks"], p2Task)     # this implicitly fills the aoilist[]
                aoilist = np.array(aoilist, dtype=[('page', 'S80'), ('id', 'S40'), ('content','S80'), ('x1',int), ('y1',int), ('x2',int), ('y2',int)])
                aoilist = aoilist.view(np.recarray)

            
            ##############################
            # AOI logging
            ##############################
            if processGazeLog:
                if not logEvents(alldata, aoilist, lastVTime, vTime, toffset):
                    # error logging, which shouldn't happen
                    logging.error("processGazeLog: error logging events for {}-{}, tOffset={}".format(lastVTime, vTime, tOffset))

            # end of AOI logging, do some updates:
            lastVTime = vTime   # used to track gazes during skimming.

            ############################
            # display video
            ############################
            if showVideo:
                if not displayFrame(windowName): break

            # console output, every 10sec video time        
            if (vTime%10000 ==0):
                print " "+str(int(vTime/10000)*10), 
                if showVideo: cv2.setTrackbarPos(taskbarName, windowName, int(frameNum/100))
        ##################
        # if flag:
        ##################
        else:
            logging.error("Error reading video frame: vt="+str(vTime)+"\tx="+str(gazex)+"\ty="+str(gazey))
            pass # no valid video frames; most likely EOF
    #########
    # no more frames
    #########
    logging.info("Ended:"+txt)
    logging.shutdown()    

def main():
    ''' Main function that processes arguments and gets things started. '''
    global yamlconfig, tess, parser, frameEngine, startFrame, showVideo, jumpAhead
    global txt, toffset, skimmingMode, logLevel, testMode, outputLogFileSuffix

    # Get command line arguments
    parser = argparse.ArgumentParser(
        description="Usage: python video2aoi.py config.yaml videofile.avi .",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        conflict_handler='resolve')
    parser.add_argument('avifiles',
                        help='The video file(s) to process.', nargs='+')
    parser.add_argument('-l', '--logLevel',
                        help='The level of informatin to be logged.',
                        choices=['INFO', 'DEBUG'],
                        default='INFO')
    parser.add_argument('-f', '--startFrame',
                        help='The frame number to start processing.', default='ignore')
    parser.add_argument('-j', '--jumpAhead',
                        help='The # of seconds to jump ahead in the skimming mode.', default='ignore')
    parser.add_argument('-o', '--offsetTime',
                        help='The msec that the video is behind the gaze timestamp.', default='ignore')
    parser.add_argument('-c', '--colorPlane',
                        help='The color plan to use for matching.',
                        choices=['ALL', 'R', 'G', 'B'],
                        default='ALL')
    parser.add_argument('-v', '--videoPlayback',
                        help='Whether to play the video or process silently.',
                        choices=['T', 'F'],
                        default='T')
    parser.add_argument('-y', '--YAMLFile',
                        default = 'default.yaml',
                        help='Name of the YAML configuration file.')
    parser.add_argument('-t', '--testingMode',
                        help='If true, no output; for testing only.',
                        choices=['T', 'F'],
                        default='F')
    parser.add_argument('-s', '--outputLogFileSuffix',
                        help='Suffix for the output log file.',
                        default='ignore')
    args = parser.parse_args()

    #################################
    # now process the args
    #################################
    showVideo = True if (args.videoPlayback is "T") else False

    yamlfile = args.YAMLFile
    try:
        yamlconfig = yaml.load(open(yamlfile))
    except:
        print "Error with the YAML file: {} cannot be opened.".format(yamlfile)
        exit(-1)
    assert "tasks" in yamlconfig
    assert "study" in yamlconfig

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
    startFrame= yamlconfig["study"]["startFrame"] if "startFrame" in yamlconfig["study"] else 0
    if args.startFrame is not 'ignore': 
        try:
            startFrame = int(args.startFrame)
        except:
            print "Error: -s {} is invalid".format(args.startFrame)
            exit(-1)

    # for skimmingMode, # of seconds to jump ahead
    jumpAhead = yamlconfig["study"]["jumpAhead"] if "jumpAhead" in yamlconfig["study"] else 0.5
    if args.jumpAhead is not 'ignore': 
        try:
            jumpAhead = float(args.jumpAhead)
        except:
            print "Error: -j {} is invalid".format(args.jumpAhead)
            exit(-1)

    # this is the async estimated by looking at video mouse movement and 
    #  cursor display based on the data from the mouse event log
    # quick hack, should be estimated automatically using template matching
    toffset = yamlconfig["study"]["videogazeoffset"] if "videogazeoffset" in yamlconfig["study"] else -600
    if args.offsetTime is not 'ignore': 
        try:
            toffset = int(args.offsetTime)
        except:
            print "Error: -o {} is invalid".format(args.offsetTime)
            exit(-1)
    
    # logfilename
    outputLogFileSuffix = yamlconfig["study"]["outputLogFileSuffix"] if "outputLogFileSuffix" in yamlconfig["study"] else "_AOI.log"
    if args.outputLogFileSuffix is not 'ignore':
        outputLogFileSuffix = args.outputLogFileSuffix

    print("outputLogFileSuffix = {}".format(outputLogFileSuffix))

    # log level is INFO unless otherwise specified
    logLevel= logging.DEBUG if "logLevelDebug" in yamlconfig["study"] and yamlconfig["study"]["logLevelDebug"] else logging.INFO
    #print "LogLevel: {} is 'DEBUG'? {}".format(args.logLevel, args.logLevel == 'DEBUG')
    if args.logLevel == 'DEBUG': 
        logLevel= logging.DEBUG
    
    # testmode
    testMode = True if (args.testingMode is "T") else False

    #print "Loglevel: INFO={} DEBUG={}; current setting is {}={}".format(logging.INFO, logging.DEBUG, args.logLevel, logLevel)
    print "INFO: startFrame: {}; jumpAhead={}; tOffset={}; logLevel={} ".format(startFrame, jumpAhead, toffset, logLevel)
    # exit(0)


    skimmingMode=False

    #################################
    # Iterate through given files
    #################################
    for vf in args.avifiles:
        for f in glob.glob(vf):
            processVideo(f)

    #################################
    # done
    #################################
    if showVideo: cv2.destroyAllWindows()
    #logging.info("End")
    logging.shutdown()
    print "end processing"

if __name__ == "__main__":

    ##########################
    # global/main:
    ###########################################################

    # txt=""
    # frame=None
    # vTime=0
    # gaze=None
    # gazex=0; gazey=0;
    
    
    main()