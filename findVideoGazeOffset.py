import numpy as np

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
        return None
    if not isinstance(videoMouseLocations, np.ndarray):
        print "Error findVideoGazeOffset(): videoMouseLocations is not a numpy array"
        return None
    if np.shape(videoMouseLocations)[0]<2 :
        print "Error findVideoGazeOffset(): need at least 2 samples for videoMouseLocations "
        return None
    if np.shape(mouseLog)[0]<2 :
        print "Error findVideoGazeOffset(): need at least 2 samples for mouseLog "
        return None
    std = np.std(np.vstack([videoMouseLocations.x, videoMouseLocations.y]).T, axis=0)
    if std[0]== 0 and std[1]==0 :
        print "Error findVideoGazeOffset(): mouse samples in videoMouseLocations cannot be all at the same location "
        return None

    # sort by time
    mouseLog.sort(order ='t')
    videoMouseLocations.sort(order='t')

    # this is a list of np arrays of indecies of matched mouse locations. 
    matchedIndices=[]
    # find candidates that matches the  mouseLocation
    for vm in videoMouseLocations:
        sqdist = np.square(mouseLog.x-vm.x) + np.square(mouseLog.y-vm.y)
        sqdist_min = np.min(sqdist)
        # if there is no match
        if sqdist_min > locationThreshold: 
            print "Info findVideoGazeOffset(): sqdist {} > {} at videoMouseLocations #{}".format(sqdist, locationThreshold, i)
            return None
        # else we find the argmins
        matchedIndices.append( np.where(sqdist == sqdist_min))


    # let's see if any of the pairs have time offset that match the observed mouse
    t0 = [m.t for m in mouseLog[matchedIndices[0]]]
    t1 = [m.t for m in mouseLog[matchedIndices[1]]]
    # now calculate a pair-wise distance matrix between the two, minus the time difference of the observed
    tdiff = np.subtract.outer(t1, t0) - (videoMouseLocations[1].t - videoMouseLocations[0].t)
    tdiff = np.square(tdiff)
    # find minimal
    tdiff_min = np.min(tdiff)

    print t0
    print t1
    print tdiff

    if tdiff_min > temporalThreshold: 
        print "Info findVideoGazeOffset(): No appropriate time difference is found, tmin = {}".format(tdiff_min)
        print tdiff
        return None 
    # now find out which rows they are:
    tMatched = np.where(tdiff == tdiff_min)
    if len(tMatched[0])>1:
        print "Info findVideoGazeOffset(): More than one match is found, tmin = {}".format(tdiff_min)
        print tdiff
        return None 
    # get the estimated gaze-video toffset
    t = t0[tMatched[0]] - videoMouseLocations[0].t
    print "Info findVideoGazeOffset(): t0[tMatched[0]]={}, videoMouseLocations[0].t={}, tOffset = {}".format(t0[tMatched[0]], videoMouseLocations[0].t, t)
    return t





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

    print findVideoGazeOffset(mouseData, testData, locationThreshold=1, temporalThreshold = 16)
