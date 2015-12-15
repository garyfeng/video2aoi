BEGIN {
    start=0
    FS="\t"
    OFS="\t"
    laptime=0
    xy[1]=-99
    xy[2]=-99
}

/KeyName:F8/ {
    # ignore everything until F8 is pressed: CamStudio StartRecording hotkey
    if(start==0) start=$3
}

/gaze/ {
    if (start>1) {
        # already started, reset time, and output only time, x, y
        laptime= $3-start
        print laptime, xy[1], xy[2]
    } 
}

/mouse move/ {
    # mouse move|27456312|(1138, 225)|0, 
    split($3, part,"|")
    
    if (start>1) {
        # already started, reset time, and output only time, x, y
        gsub(/[\(\) ]/, "", part[3])
        split(part[3], xy, ",")
        # mouse info too dense. Only print at the frequence of gaze; see above
        #print part[2]-start, xy[1], xy[2]
    } else {
        # do nothing
    }
}

