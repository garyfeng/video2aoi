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
	x=$9; y=$11
	info = $13
	page = $15; gsub(/Assessment\/items\//, "", page)
	aoi = $16
	content = $17
	print subj, "g", t, x, y, info, page, aoi, content, $18, $19, $20, $21
}

/keyboard:/ {
	# INFO	397465	Keystroke: vt=992000.0	gzt=990297	x=	y=	key=s
	# note the space after MOuse:, it's a bug that is fixed in teh next version; should have been \t
	# this will change the parsing
	t=$5+toffset
	#page= $15; gsub(/Assessment\/items\//, "", page)
	# use the current page info
	print subj, "k", t, "", "", "", page, "", $13, "", "", "", ""
}

/mouse:/ {
    #INFO	6556006	mouse:	vt=731758	gzt=730258	x=1858	y=480	info=0	aoi=Assessment/items/MSQ7						
	# note the space after MOuse:, it's a bug that is fixed in teh next version; should have been \t
	# this will change the parsing
	t=$5+toffset
	x=$9; y=$11
	info = $13
	page= $15; gsub(/Assessment\/items\//, "", page)
	aoi=$16
	content = $17
	print subj, "m", t, x, y, info, page, aoi, content, "", "", "", ""
}
