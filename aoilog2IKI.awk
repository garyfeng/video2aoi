# Gary Feng, 2013
# copyleft

# this script reads the *AOI.log file and prints out 
# format: Subj, keycounter, tLastKey, tCurrKey, prevIKI, keycode, AOI_name, timeOnAOI, %TimeOnAOI
# note that the timeOnAOI may be longer than the prevIKI, because the key presses may happen in between 
#    gaze samples, and sometimes the gaze samples don't come in regular times (particularly for Tobii).
# but overall these should be very close

BEGIN {
    start=0
    FS="[=\t]"
    OFS="\t"
    subj = ""
	page = ""
	aoi=""
	content =""
	counter = -1
	# vt now is back-calcualted from gzt taking into account of the toffset
	# no need to do this anymore
	toffset = 0
	lastaoi = "sdfsdfa;ljslfjs;j"
	lastpage = ""
	lastcontent =""
	stime=-9999
	etime=0
	tlastgaze = 0

	init()

}


function init() {
	# initialize the array
	# commented code reserved to limit the # of AOIs 
	for (i in aoilist) delete aoilist[i]
	# aoilist["Task4AdLeft"]=0
	# aoilist["Task4Checklist"]=0
	# aoilist["EssayDirections"]=0
	# aoilist["EssayInput"]=0
	# aoilist["Task4FeedbackSubmitted"]=0
	# aoilist["Task4PlanOutlineFullScreen"]=0
	# aoilist["Task4PlanIdeaWebFullScreen"]=0
	# aoilist["Task4PlanListFullScreen"]=0
	# aoilist["Task4PlanIdeaTreeFullScreen"]=0
	# aoilist["Task4PlanFreeWritingFullScreen"]=0
	# aoilist["Task4PlanPreviewFullScreen"]=0
	# aoilist["Task4PlanPreviewLeft"]=0
	# aoilist["Task4TryingFullScreen"]=0
	# aoilist["Task4TryingLeft"]=0
	# aoilist["Task4WorriesFullScreen"]=0
	# aoilist["Task4WorriesLeft"]=0
	# aoilist["Task4TipsRight"]=0
	# aoilist["Assessment"]=0
	# aoilist["MISSING"]=0
	# aoilist["JUNK"]=0
}

/Scaling ratio/ {
	# INFO	970	video = cbalm08.avi	Scaling ratio =0.5	log = 'C:\Users\gfeng\Documents\My Projects\2013 CBAL Math\video2aoi\eyelink\cbalm08.avi_AOI.log'
	subj = $4	#cbalm08.avi
	gsub(/\.avi/, "", subj)	#cbalm08
}

/gaze:/{
	#INFO	41315	gaze:	vt=77104	gzt=75604	x=936	y=696	info=4197	aoi=Assessment/items/Task4Intro/IntroDirections	IntroDirections	IntroDirections	569	502	1351	726
	# header
	# subj	event	t	x	y	sampleNum	page	aoi	content	x1	y1	x2	y2
	t=$5 + toffset
	page = $15
	aoi = $16
	content = $17

	if (aoi=="Assessment") aoi="Non-AOI"
	if (aoi=="") aoi="Non-AOI"

	if ($9== -32768) {
		aoilist["MISSING"] += t-tlastgaze
	} else {
		aoilist[aoi] += t-tlastgaze
	}

	# update
	tlastgaze =t

	# if (aoi in aoilist) {
	# 	# legit aoi to count
	# 	aoilist[aoi] +=1
	# } else 
	# 	aoilist["JUNK"] +=1
	# }
}

/keyboard:/ {
	# INFO	397465	Keystroke:	vt=992000.0	gzt=990297	x=	y=	key=s

	t=$5
	#lead = sprintf ("%s\t%s\t%d\t%d\t%d\t%d\t%s", subj, page, ++counter, stime, t, t-stime, $13)
	# page is useless because we have the unique aoi
	# format: Subj, keycounter, tLastKey, tCurrKey, prevIKI, keycode, AOI_name, timeOnAOI, %TimeOnAOI
	lead = sprintf ("%s\t%d\t%d\t%d\t%d\t%s", subj, ++counter, stime, t, t-stime, $13)

	# calc the total aoi time to print % for each aoi
	totaltime = 0
	for (a in aoilist) totaltime += aoilist[a]
	# make sure aoilist is not completely empty, or else you don't have an output
	if (totaltime==0) aoilist["MISSING"]=0

	# print sorted
	n=asorti(aoilist, ind)
	for (i=1; i<=n; i++) {
		if (totaltime ==0) aoiTimePrcnt="1.000"
		else aoiTimePrcnt = sprintf("%04.3f", aoilist[ind[i]]/totaltime)
		if(stime > 0) print lead, ind[i], aoilist[ind[i]], aoiTimePrcnt
	}
	#for (a in aoilist) print "AOI", a, aoilist[a]

	# update
	stime=t
	init()
}
