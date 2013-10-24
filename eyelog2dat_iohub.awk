# gary feng, 2013
# princeton nj

# awk script to turn iohub _eye.log output into a fromat that 
# video2aoi can read in and process
# note that there is only gaze information currently
# will design a log file with eye and keystroke/mouse info, too

BEGIN {
    start=0
    FS="\t"
    OFS="\t"
    laptime=0
    xy[1]=-99
    xy[2]=-99
}

/.*/ {
    # data format for the iohub _eye.log
    # x, y, t
    if (NR==1) {start=$3}
    print int(($3-start)*1000), "gaze", int($1), int($2), NR
}
