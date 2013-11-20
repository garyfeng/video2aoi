BEGIN {
    start=0
    FS="[=\t]"
    OFS="\t"
    subj = ""
	page = ""
	aoi=""
	content =""
	# vt now is back-calcualted from gzt taking into account of the toffset
	# no need to do this anymore
	toffset = 0
	lastaoi = "sdfsdfa;ljslfjs;j"
	lastpage = ""
	lastcontent =""
	stime=-9999
	etime=0
}

/Scaling ratio/ {
	# INFO	970	video = cbalm08.avi	Scaling ratio =0.5	log = 'C:\Users\gfeng\Documents\My Projects\2013 CBAL Math\video2aoi\eyelink\cbalm08.avi_AOI.log'
	subj = $4	#cbalm08.avi
	gsub(/\.avi/, "", subj)	#cbalm08
}

/gaze:|\tGaze\t/{
	#INFO	41315	gaze:	vt=77104	gzt=75604	x=936	y=696	info=4197	aoi=Assessment/items/Task4Intro/IntroDirections	IntroDirections	IntroDirections	569	502	1351	726
	# header
	# subj	event	t	x	y	sampleNum	page	aoi	content	x1	y1	x2	y2
	t=$5 + toffset
	page = $15
	aoi = $16
	content = $17

	# ignore several aois that are double-coded so meaningless but serve to break up the streak below:
	if (aoi ~ /LSIDE$|RSIDE$|TOP$/) next
	# Q10 has many overlapped aois
	if (aoi ~ /MSQ1TOTALIMAGE|MSQ10STEM|MSQ10GRAPH|MSQ10Q&RA[A-E]|MSQ10COMBINEBUTTON|MSQ10ANN/) next
	# simulation has overlapping aois
	if (aoi ~/Simulation|MSPRACTICETOOLBUTTON|MSPRACTICECALCULATORBUTTON|MSQ8UNPLAYEDANIMATIO/) next
	# Q3 graph and buttons inside
	if (aoi ~/MSQ3LINEBOX|MSQ3UNDOBOX|MSQ3STARTOVERBOX|MSTITLEBANNER|MSTITLENAVIGATION/) next

	#skip over non-AOI samples if they are < n msec (3 samples) since the last known AOI
	# @@ logic not correct: stime is from the last AOI onset, not the start of the "" AOI.
	#if (aoi =="" && t-stime<100) next
	
	#if (aoi=="Assessment" && content = "Assessment") next
	# if missing data, then count toward the last aoi
	if ($9 ==-32768 ) aoi = "MISSING"
	#if ($9 ==-32768 && t-stime<=100) next

	# now simplify a few page titles that involves hierarchies
	# if (page ~ /Assessment\/items\/Task4EssayRight/) page = "Assessment/items/Task4EssayRight"
	# if (page ~ /Assessment\/items\/Task4Intro/) page = "Assessment/items/Task4Intro"
	# if (page ~ /TypingTestResults\/TypingResults/) page = "TypingTestResults"
	gsub( /Assessment\/items\//, "", page)
	# only print if the AOI has changed
	if (aoi != lastaoi) {
		# this line is a new aoi, print the last aoi
		if (stime>0) {
			# only print if this is not the very first line

			print subj, stime, t, t-stime, lastpage, lastaoi, lastcontent, lastx1, lasty1, lastx2, lasty2
		}
		stime = t
		etime = t
		lastaoi = aoi
		lastpage = page
		lastcontent = content
		lastx1 = $18; lastx2=$20; lasty1 = $19; lasty2=$21
	}
	#print subj, "g", t, $9, $11, $13, $15, $16, $17, $18, $19, $20, $21
}

