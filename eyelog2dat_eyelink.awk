BEGIN {
    start=0
    FS="\t"
    OFS="\t"
    laptime=0
    xy[1]=-99
    xy[2]=-99
    gazeCount = 0
}


# gary feng
function gettime(str)
{
    # str = 2013-03-12 09:36:03,375
    split(str, dump," ")
    # get the time part
    str = dump[2]
    split(str, dump, ":")
    # get ms 
    split(dump[3], d2, ",")

    return (int(dump[1])*3600 + int(dump[2]*60)+int(d2[1]))*1000 + int(d2[2])
}


/screen_pixel_coords/ {
    # this is when the calibration began
    calibStart = gettime($1)
    if(start==0) start = calibStart
}

/EyeTracker.startRecording:/ {
    # this is when calibration was over
    calibEnd = gettime($1)
}

# END {
#     # now let's figure out the delay
#     # 2013-03-12 09:36:03,375 info    EyeTracker.calibration: completed
#     # 2013-03-12 09:40:45,467 info    EyeTracker.startRecording: 

#     print FILENAME, (calibEnd- calibStart)

# }


# /KeyName:F8/ {
#     # ignore everything until F8 is pressed: CamStudio StartRecording hotkey
#     if(start==0) start=$3
# }

/gaze/ {
    if (start>1) {
        # already started, reset time, and output only time, x, y
        #laptime= $3-start
        laptime= gettime($1)-start
        gazeCount = gazeCount+1
        print laptime, "gaze", $4, $5, gazeCount
    } 
}

/keyboard/ {
    if (start>1) {
        # already started, reset time, and output only time, x, y
        laptime= gettime($1)-start
        print laptime, "keyboard", "", "", $5
    } 
}

/info/ {
    # info doesn't have timing, use the last known timed event
    # no x, y either
    print laptime, "info", "", "", $3
}

/mouse move/ {
    laptime= gettime($1)-start
    # mouse move|27456312|(1138, 225)|0, 
    split($4, part,"|")
    if (start>1) {
        # already started, reset time, and output only time, x, y
        gsub(/[\(\) ]/, "", part[3])
        split(part[3], xy, ",")
        # mouse info too dense. Only print at the frequence of gaze; see above
        print laptime, "mouseMove", xy[1], xy[2],""
    } else {
        # do nothing
    }
}

/mouse left down/ {
    laptime= gettime($1)-start
    # mouse left down|183692671|(607, 804)|0,  
    split($4, part,"|")
    
    if (start>1) {
        # already started, reset time, and output only time, x, y
        gsub(/[\(\) ]/, "", part[3])
        split(part[3], xy, ",")
        #print part[2]-start, "mouseClick", xy[1], xy[2],""
        print laptime, "mouseClick", xy[1], xy[2],""
    } else {
        # do nothing
    }
}
