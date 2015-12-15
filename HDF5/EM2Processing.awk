BEGIN {
    start=0
    FS="\t"
    OFS="\t"
}

function date2ms(date) {
	#2012-09-05 13:30:16,632
    split(date, x, "[ ,:]")
    t=x[2]*3600*1000+x[3]*60*1000+x[4]*1000+x[5]
	return (t)
}


/screen_pixel_coords/ {
    # ignore everything until F8 is pressed: CamStudio StartRecording hotkey
    if(start==0) start=date2ms($1)
}

/gaze/ {
    if (start>1) {
        # already started, reset time, and output only time, x, y
        print date2ms($1)-start, $4, $5
    } else {
        # do nothing
    }
}

