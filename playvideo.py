import cv2
import cv2.cv as cv
import os
import os.path
import sys
import numpy as np
#import atexit
import glob

video=None

def onChange (c):
    global video
    video.set(cv.CV_CAP_PROP_POS_FRAMES, c*30)
    
def playVideo(v):
    '''Process a video file, and log the events and AOIs to the log file.
    It uses the global settings from the yamlconfig object. 
    '''
    global video
    
    try:
        video = cv2.VideoCapture(v)
    except:
        logging.error("Error: cannot open video file "+v)
        print "Error: cannot open video file "+v
        return
    # close all windows, then open this one
    cv2.destroyAllWindows()
    windowName=v
    cv2.namedWindow(v)
    taskbarName="Video Control"
    startFrame=1
    nFrames = int( video.get(cv.CV_CAP_PROP_FRAME_COUNT ))
    cv2.createTrackbar(taskbarName, windowName, int(startFrame/30), int(nFrames/30+1), onChange)
    # get the video

        # now loop through the frames
    while video.grab():
        frameNum = video.get(cv.CV_CAP_PROP_POS_FRAMES)
        
        flag, frame = video.retrieve()
        if flag:
            text_color = (128,128,128)
            txt = v+"\t"+str(frameNum)

            levels=60
            minc=15; maxc=255
            bins = np.arange(levels).reshape(levels,1)
            color = [ (0,0,255),(0,255,0),(255,0,0) ]
            for ch, col in enumerate(color):
                hist_item = cv2.calcHist([frame],[ch],None,[levels],[minc, maxc])
                cv2.normalize(hist_item,hist_item,minc, maxc,cv2.NORM_MINMAX)
                hist=np.int32(np.around(hist_item))
                pts = np.column_stack((bins*8,1000-hist))
                cv2.polylines(frame,[pts],False,col)

            cv2.putText(frame, txt, (20,50), cv2.FONT_HERSHEY_PLAIN, 1.0, text_color, thickness=1)
            
            # resize
            frame = cv2.resize(frame, (0,0), fx= 0.5, fy=0.5)
            cv2.imshow(windowName, frame)       # main window with video control
            #if txtScrollImage is not None: cv2.imshow("txtScrollImage", txtScrollImage)
            #if txtBitmap is not None: cv2.imshow("txt", txtBitmap)
                
            # keyboard control; ESC = quit
            key= cv2.waitKey(1) #key= cv2.waitKey(waitPerFrameInMillisec)
            if (key==27):
                print("Key: ESC pressed")
                exit(0)

if __name__ == "__main__":

    ##########################
    # global/main:
    ###########################################################
    # Starting
    v="*.avi"
    # video input filename
    if(len(sys.argv)==2):
        v=sys.argv[1]
    else:
        print "USAGE: playvideo.py videoname.avi"
        exit(0)
    playVideo(v)
