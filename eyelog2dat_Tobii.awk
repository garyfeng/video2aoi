BEGIN {
    start=0
    FS="\t"
    OFS="\t"
    laptime=0
    xy[1]=-99
    xy[2]=-99
    subj=""
}

/Participant/ {
    subj = $2
}
    
/ScreenRecStarted/ {
    # This is the Tobii TSV signal for video starting
    # time is always $1
    if(start==0) start=$1
}

/^[0-9]+/ {
    # data line
    if (start>1) {
        # already started, reset time, and output only time, x, y
        laptime= $1-start
        if (length($20)>0) {
            print laptime, "gaze", $20, $21, ""
        } else if ($22=="KeyPress") {
            print laptime, "keyboard", "", "", $26
        } else if ($22=="LeftMouseClick") {
            print laptime, "mouseClick", $24, $25, $22
        } else if (length($22)>0) {
            print laptime, "info", "", "", $22+" "+$23+" "+$24+" "+$25+" "+$26
        } 
    } 
}

/^[a-zA-Z]+/ {
    gsub(/\t/, "   ")
    print laptime, "info", "", "", $0
}


