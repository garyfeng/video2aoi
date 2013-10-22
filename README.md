video2aoi
=========
Video2AOI is a collection of python scripts to define and tag Areas-of-interest (AOIs) for 
screen-casting videos recorded during eye-tracking studies. It takes a YAML definition file
and a screen-casting video (recommended H264 turned for stillimage). The output is typically
a log file with various pieces of information in it, depending on how you set up the YAML
configeration file.

USAGE: video2aoi.py config.yaml video.avi


Dependencies:

-- OpenCV 2.4 for Python 2.7
-- Numpy
-- Tesseract for Python 2.7
-- YAML for python
