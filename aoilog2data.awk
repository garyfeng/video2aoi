BEGIN {
    start=0
    FS="[=\t]"
    OFS="\t"
    subj = ""
}

/Scaling ratio/ {
	# INFO	970	video = cbalm08.avi	Scaling ratio =0.5	log = 'C:\Users\gfeng\Documents\My Projects\2013 CBAL Math\video2aoi\eyelink\cbalm08.avi_AOI.log'
	subj = $4	#cbalm08.avi
	gsub(/\.avi/, "", subj)	#cbalm08
}

/gzt/{
	#INFO	44480	Gaze	vt=615879	gzt=614579	x=784	y=438	info=1031	aoi=Assessment/items/MSTITLE	MSTITLEPARTII	MSTITLEPARTII	657	397	1068	461
	# header
	# subj	t	x	y	sampleNum	page	aoi	content	x1	y1	x2	y2
	print subj, $5, $9, $11, $13, $15, $16, $17, $18, $19, $20, $21
}

