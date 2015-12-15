BEGIN {
    start=0
    FS="\t"
    OFS="\t"
    laptime=0
    xy[1]=-99
    xy[2]=-99
    count=0
}

/KeyName:F8/ {
    # ignore everything until F8 is pressed: CamStudio StartRecording hotkey
    if(start==0) start=$3
}


/mouse left down/ {
    # mouse left down|183692671|(607, 804)|0,  
    split($3, part,"|")
    
    if (start>1) {
        # already started, reset time, and output only time, x, y
        gsub(/[\(\) ]/, "", part[3])
        split(part[3], xy, ",")
        x=xy[1]
        y=xy[2]
        # only print if the coords are in the NEXT button AND the time between 2 clicks is >1000msec
        if (x<1345 && x>(1345-72) && y<267 && y>(267-54) && $2-lasttime>1000) {
            count=count+1
            #print "page=" count, "time=" $2-start, "x=" x, "y=" y
            print count, $2-start
            lasttime=$2
        }
    } else {
        # do nothing
    }
}
