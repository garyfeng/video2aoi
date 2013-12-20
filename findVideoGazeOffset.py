import numpy as np
import logging

#from FrameEngine import *    #FrameEngine
#from video2aoi import displayFrame, p2ReadSignatureImage

#global yamlconfig

def findGazeVideoOffset(mouseLog, videoMouseLocations, locationThreshold = 2, temporalThreshold = 250):
    '''Given the mouseLog (in numpy array) and videoMouseLocations (a numpy array) that contains
    some fractions of mouse locations found in the video frames via template matching, find the 
    most likely time shift parameter. None if no offset parameter can be found. 

    locationThreshold is the parameter for difference in x,y locations between the mouseLog and 
    videoMouseLocations; default is 2 pixels in either x or y.

    temporalThreshold is the parameter for the search window. 

    '''
    logging.getLogger('')

    if not isinstance(mouseLog, np.ndarray):
        logging.error( "Error findGazeVideoOffset(): mouseLog is not a numpy array")
        return None
    if not isinstance(videoMouseLocations, np.ndarray):
        logging.error( "Error findGazeVideoOffset(): videoMouseLocations is not a numpy array")
        return None
    if np.shape(videoMouseLocations)[0]<2 :
        logging.error( "Error findGazeVideoOffset(): need at least 2 samples for videoMouseLocations ")
        return None
    if np.shape(mouseLog)[0]<2 :
        logging.error( "Error findGazeVideoOffset(): need at least 2 samples for mouseLog ")
        return None
    std = np.std(np.vstack([videoMouseLocations.x, videoMouseLocations.y]).T, axis=0)
    if std[0]== 0 and std[1]==0 :
        logging.error( "Error findGazeVideoOffset(): mouse samples in videoMouseLocations cannot be all at the same location ")
        return None
    std = np.std(np.vstack([mouseLog.x, mouseLog.y]).T, axis=0)
    if std[0]== 0 and std[1]==0 :
        logging.error( "Error findGazeVideoOffset(): mouse samples in mouseLog cannot be all at the same location ")
        return None

    # sort by time
    mouseLog.sort(order ='t')
    videoMouseLocations.sort(order='t')

    # this is a list of np arrays of indecies of matched mouse locations. 
    matchedIndices=[]
    # find candidates that matches the  mouseLocation
    #print "videoMouseLocations={}".format (videoMouseLocations)
    for vm in videoMouseLocations:
        #print "vm = {}".format(vm)
        # vm is a tuple
        sqdist = np.square(mouseLog.x-vm[1]) + np.square(mouseLog.y-vm[2])
        sqdist_min = np.min(sqdist)
        # if there is no match
        if sqdist_min > locationThreshold: 
            logging.info( "findGazeVideoOffset(): sqdist {} > {} at videoMouseLocations #{}".format(sqdist_min, locationThreshold, vm))
            print "findGazeVideoOffset(): sqdist {} > {} at videoMouseLocations #{}".format(sqdist_min, locationThreshold, vm)
            return None
        # else we find the argmins, convert to list
        matchedIndices.append( np.where(sqdist == sqdist_min))
    logging.debug( "matchedIndices = {}".format(matchedIndices))
    #print "matchedIndices = {}".format(matchedIndices)

    # drop those locations with multiple matches, i.e., where the mouse has passed several times
    # so what's left is the unique matches
    #sumV=0; sumG=0; 
    c=0; 
    lastInd=0;
    tList = []
    for i in range(len(matchedIndices)):
        indList = matchedIndices[i][0].tolist()
        # eliminate ones that are too far apart or going backwards. 
        #if len(indList) ==1 and lastInd<indList[0] and indList[0]-lastInd<1000:
        if len(indList)>0 and lastInd<indList[0] and indList[0]-lastInd<1000:
            # # good out-of-order cases ... impossible when we are processing frames one by one
            # sumG += mouseLog.t[indList[0]]
            # sumV += videoMouseLocations.t[i]
            c +=1

            tList.append(mouseLog.t[indList[0]]-videoMouseLocations.t[i])

            logging.info('findGazeVideoOffset: #{} index={} | GazeTime={} VideoTime = {} dT={}'.format(
                c, indList[0], mouseLog.t[indList[0]], videoMouseLocations.t[i],
                mouseLog.t[indList[0]]-videoMouseLocations.t[i]))
            #print 'findGazeVideoOffset: #{} index={} | GazeTime={} VideoTime = {} dT={}'.format(
            #    c, indList[0], mouseLog.t[indList[0]], videoMouseLocations.t[i],
            #    mouseLog.t[indList[0]]-videoMouseLocations.t[i])
            lastInd = indList[0]

    # now estimate the offset by taking an average of all
    # if c>0: 
    #     t = (sumG- sumV)/c
    # else:
    #     t = None   
    # # using a different algorithm, throwing out some extreme values
    tList.sort()
    print tList
    logging.debug("findGazeVideoOffset\t time differences = {}".format(tList))

    t=None
    if len(tList)>7:
        tList=tList[2:-2]   # remove the max and min
        logging.debug("findGazeVideoOffset\t using {}".format(tList))
        t=sum(tList)/len(tList)
    # if time offset is off by more than 5 seconds, then something is definitely wrong. Keep searching
    if t is not None and abs(t)>15000: 
        logging.info('findGazeVideoOffset: estimated t={} >|15000|'.format(t))
        print 'findGazeVideoOffset: estimated t={} >|15000|'.format(t)
        #t=None
    #logging.info("findGazeVideoOffset\t#samples={} sumG={} sumV={} offset={}".format(c, sumG, sumV, t))

    #print "t= {}".format(t)
    logging.info("findGazeVideoOffset:\t{}".format(t))
    return t


    ########################
    # the following is an alternative algorithm for estimating the toffset
    # not fully debugged and probably unnecessary
    ########################
    # # notice we make it a plain list of integers
    # matchedIndices = [i[0] for i in matchedIndices if len(i)==1]
    # if len(matchedIndices) <1:
    #     # not unique enough data
    #     return None

    # # now get a list of the timestamps of these unique matches
    # tm=mouseLog[matchedIndices].t



    # # let's see if any of the pairs have time offset that match the observed mouse
    # t0=mouseLog[matchedIndices[0]].t
    # t1=mouseLog[matchedIndices[1]].t
    # #t0 = [m.t for m in mouseLog[matchedIndices[0]]]
    # #t1 = [m.t for m in mouseLog[matchedIndices[1]]]
    # # now calculate a pair-wise distance matrix between the two, minus the time difference of the observed
    # tdiff = np.subtract.outer(t1, t0) - (videoMouseLocations.t[1] - videoMouseLocations.t[0])
    # tdiff = np.absolute(tdiff)
    # # find minimal
    # tdiff_min = np.min(tdiff)

    # logging.info( "t0={}".format(t0))
    # logging.info( "t1={}".format(t1))
    # logging.info( "tdiff={}".format(tdiff))
    # logging.info( "tdiff_min={}".format(tdiff_min))
    # print "t0={}".format(t0)
    # print "t1={}".format(t1)
    # print "tdiff={}".format(tdiff)
    # print "tdiff_min={}".format(tdiff_min)

    # if tdiff_min > temporalThreshold: 
    #     logging.info( "Info findGazeVideoOffset(): No appropriate time difference is found, tmin =  {} > {}".format(tdiff_min, temporalThreshold))
    #     return None 
    # # now find out which rows they are:
    # tMatched = np.where(tdiff == tdiff_min)
    # print "tMatched = {}".format(tMatched)

    # if len(tMatched[0])==0:
    #     logging.info( "Info findGazeVideoOffset(): This shouldn't happen: no match is found, tdiff = {} tdiff_min {}".format(tdiff, tdiff_min))
    #     return None 
    # if len(tMatched[0])>1:
    #     logging.info( "Info findGazeVideoOffset(): More than one match is found, tmin = {} > {}".format(tdiff_min, temporalThreshold))
    #     return None 
    # # get the estimated gaze-video toffset
    # # note that np.where returns the (y,x) coordinate of the minimals
    # # the y axis correspond to t1, and x to t0 in our case
    # # we use the t1 for calculation; we can easily use t0 if we want to.
    # print "t1[tMatched[0]] - videoMouseLocations.t[1] = {} - {}".format(t1[tMatched[0]], videoMouseLocations.t[1])

    # t = t1[tMatched[0]] - videoMouseLocations.t[1]
    # t=t[0]
    # logging.info( "Info findGazeVideoOffset(): t1[tMatched[0]]={}, videoMouseLocations[1].t={}, tOffset = {}".format(t1[tMatched[0]], videoMouseLocations.t[1], t))
    # return t





if __name__ == "__main__":
    # unit testing
    mouseDataList = [
        (669408, "mouse", 758, 528, 0),
        (669432, "mouse", 759, 529, 0),
        (669464, "mouse", 760, 530, 0),
        (669528, "mouse", 760, 530, 0),
        (669598, "mouse", 760, 531, 0),
        (669617, "mouse", 760, 531, 0),
        (669648, "mouse", 760, 532, 0),
        (669672, "mouse", 760, 533, 0),
        (669688, "mouse", 760, 533, 0),
        (669711, "mouse", 760, 534, 0),
        (669735, "mouse", 760, 534, 0),
        (669866, "mouse", 760, 535, 0),
        (669873, "mouse", 760, 535, 0),
        (670033, "mouse", 760, 536, 0),
        (670103, "mouse", 760, 537, 0),
        (670119, "mouse", 760, 537, 0),
        (670174, "mouse", 760, 538, 0),
        (670286, "mouse", 760, 538, 0),
        (670327, "mouse", 760, 539, 0),
        (670390, "mouse", 760, 540, 0),
        (670408, "mouse", 760, 540, 0),
        (670557, "mouse", 760, 541, 0),
        (670816, "mouse", 760, 541, 0),
        (673240, "mouse", 760, 542, 0),
        (673998, "mouse", 760, 542, 0),
        (674014, "mouse", 760, 543, 0),
        (674055, "mouse", 759, 544, 0),
        (674095, "mouse", 759, 529, 0),
        (674104, "mouse", 758, 528, 0),
        (674112, "mouse", 760, 548, 0),
        (674122, "mouse", 761, 551, 0),
        (674126, "mouse", 763, 552, 0),
        (674134, "mouse", 763, 554, 0),
        (674142, "mouse", 764, 556, 0),
        (674150, "mouse", 765, 557, 0),
        (674158, "mouse", 765, 559, 0),
        (674167, "mouse", 765, 564, 0),
        (674175, "mouse", 765, 566, 0),
        (674184, "mouse", 765, 567, 0),
        (674191, "mouse", 765, 568, 0)
        ]

    testDataList = [
        (667408, "mouse", 758, 528, 0),
        (667434, "mouse", 759, 529, 0),
        (667598, "mouse", 760, 531, 0)
        ]

    # to import to rec array, you have to transpose it.
    mouseDataListT = [[i[0] for i in mouseDataList], 
        [i[1] for i in mouseDataList], 
        [i[2] for i in mouseDataList], 
        [i[3] for i in mouseDataList], 
        [i[4] for i in mouseDataList]]

    testDataListT = [[i[0] for i in testDataList], 
        [i[1] for i in testDataList], 
        [i[2] for i in testDataList], 
        [i[3] for i in testDataList], 
        [i[4] for i in testDataList]]

    mouseData = np.core.records.fromarrays(mouseDataListT, names=['t', 'event', 'x', 'y', 'info'])

    testData = np.core.records.fromarrays(testDataListT, names=['t', 'event', 'x', 'y', 'info'])    # change the data above in various ways to test the algorithm;
    # change the x,y 
    # change the timing

    print findGazeVideoOffset(mouseData, testData, locationThreshold=1, temporalThreshold = 16)
