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

/gaze:/{
	#INFO	41315	gaze:	vt=77104	gzt=75604	x=936	y=696	info=4197	aoi=Assessment/items/Task4Intro/IntroDirections	IntroDirections	IntroDirections	569	502	1351	726
	# header
	# subj	event	t	x	y	sampleNum	page	aoi	content	x1	y1	x2	y2
	t=$5 + toffset
	page = $15
	aoi = $16
	content = $17
	print subj, "g", t, $9, $11, $13, $15, $16, $17, $18, $19, $20, $21
}

/keyboard:/ {
	# INFO	397465	Keystroke: vt=992000.0	gzt=990297	x=	y=	key=s
	# note the space after MOuse:, it's a bug that is fixed in teh next version; should have been \t
	# this will change the parsing
	t=$5+toffset
	print subj, "k", t, "", "", "", page, "", $13, "", "", "", ""
}

/mouse:/ {
#logging.info("Mouse:\tvt="+str(vTime)+"\tgzt="+str(mousetime)+"\tx="+str(mousex)+"\ty="+str(mousey)+"\tkey="+str(i["info"]))
	# note the space after MOuse:, it's a bug that is fixed in teh next version; should have been \t
	# this will change the parsing
	t=$5+toffset
	print subj, "m", t, $9, $10, $11, page, "", $11, "", "", "", ""
}
