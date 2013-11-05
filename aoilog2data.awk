BEGIN {
    start=0
    FS="[=\t]"
    OFS="\t"
    subj = ""
	page = ""
	aoi=""
	content =""
	toffset = 1500
}

/Scaling ratio/ {
	# INFO	970	video = cbalm08.avi	Scaling ratio =0.5	log = 'C:\Users\gfeng\Documents\My Projects\2013 CBAL Math\video2aoi\eyelink\cbalm08.avi_AOI.log'
	subj = $4	#cbalm08.avi
	gsub(/\.avi/, "", subj)	#cbalm08
}

/\tGaze\t/{
	#INFO	44480	Gaze	vt=615879	gzt=614579	x=784	y=438	info=1031	aoi=Assessment/items/MSTITLE	MSTITLEPARTII	MSTITLEPARTII	657	397	1068	461
	# header
	# subj	event	t	x	y	sampleNum	page	aoi	content	x1	y1	x2	y2
	t=$7 + toffset
	page = $15
	aoi = $16
	content = $17
	print subj, "g", t, $9, $11, $13, $15, $16, $17, $18, $19, $20, $21
}

/Keystroke:/ {
	# INFO	397465	Keystroke: vt=992000.0	gzt=990297	x=	y=	key=s
	# note the space after MOuse:, it's a bug that is fixed in teh next version; should have been \t
	# this will change the parsing
	t=$6+toffset
	print subj, "k", t, "", "", "", page, "", $12, "", "", "", ""
}

/Mouse:/ {
#logging.info("Mouse: vt="+str(vTime)+"\tgzt="+str(mousetime)+"\tx="+str(mousex)+"\ty="+str(mousey)+"\tkey="+str(i["info"]))
	# note the space after MOuse:, it's a bug that is fixed in teh next version; should have been \t
	# this will change the parsing
	t=$6+toffset
	print subj, "m", t, $8, $9, $10, page, "", $10, "", "", "", ""
}
