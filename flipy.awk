# gary feng, 2013
# princeton nj

# awk script to fix a bug in the iohub data export
# where the y axis shoudl be flipped when data are center-origined. 

BEGIN {
    start=0
    FS="\t"
    OFS="\t"
}

{
    # data format for the iohub _eye.log
    # x, y, t
    if ($4 != "") $4 = 1280-$4 - 100
    # t, "gaze", x, y, info
    print $0
}
