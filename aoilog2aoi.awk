# Gary Feng, 2013 Princeton, NJ
# script to filter AOIDAT info from aoi logs and to convert to nVivo format
# which is 00:01:23.3 \t text

# gary feng
function msec2minutes(msec)
{
    hour = int(msec/3600000)
    msec = msec % 3600000
    minute = int(msec/60000)
    msec = msec % 60000
    second = int(msec/1000)
    msec = msec % 1000

    t = sprintf ("%02i:%02i:%02i.%03i", hour, minute, second, msec)

    return t

}

BEGIN {
    start=0
    FS="[=\t]"
    OFS="\t"
    subj = ""
    lastAOI = ""
}

/Scaling ratio/ {
	# INFO	970	video = cbalm08.avi	Scaling ratio =0.5	log = 'C:\Users\gfeng\Documents\My Projects\2013 CBAL Math\video2aoi\eyelink\cbalm08.avi_AOI.log'
	subj = $4	#cbalm08.avi
	gsub(/\.avi/, "", subj)	#cbalm08
	subj = subj "_aoi.txt"
}

/AOIDAT/{
	# INFO	4861	AOIDAT		video='fbs08.avi'	t=403367.0	frame=12101.0	Assessment	TOOLBARLEFT	450	245	758	312	TOOLBARLEFT
	thisAOI = $11
	printThis = 1

	#print only if the following are NOT true
	if (thisAOI == lastAOI) 	printThis =0
	if (length(thisAOI)==0) 	printThis =0
	if (thisAOI=="Assessment") 	printThis =0

	# now let's see if we print
	if (printThis ==1) {
		print msec2minutes($8),  thisAOI > subj
		lastAOI = thisAOI
	}
}

