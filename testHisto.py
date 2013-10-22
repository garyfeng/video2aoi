# http://subversion.assembla.com/svn/imqual/fichiers_exemple/metrics/testHisto.py

# Calculating and displaying 2D Hue-Saturation histogram of a color image
import math
import sys
import cv2.cv as cv

#import scipy
import numpy
#from scipy import fftpack
import random


def hs_histogram(src):
    # Convert to HSV
    hsv = cv.CreateImage(cv.GetSize(src), 8, 3)
    cv.CvtColor(src, hsv, cv.CV_BGR2HSV)

    # Extract the H and S planes
    h_plane = cv.CreateMat(src.rows, src.cols, cv.CV_8UC1)
    s_plane = cv.CreateMat(src.rows, src.cols, cv.CV_8UC1)
    cv.Split(hsv, h_plane, s_plane, None, None)
    planes = [h_plane, s_plane]

    h_bins = 30
    s_bins = 32
    hist_size = [h_bins, s_bins]
    # hue varies from 0 (~0 deg red) to 180 (~360 deg red again */
    h_ranges = [0, 180]
    # saturation varies from 0 (black-gray-white) to
    # 255 (pure spectrum color)
    s_ranges = [0, 255]
    ranges = [h_ranges, s_ranges]
    scale = 10
    hist = cv.CreateHist([h_bins, s_bins], cv.CV_HIST_ARRAY, ranges, 1)
    cv.CalcHist([cv.GetImage(i) for i in planes], hist)
    (_, max_value, _, _) = cv.GetMinMaxHistValue(hist)

    hist_img = cv.CreateImage((h_bins*scale, s_bins*scale), 8, 3)

    for h in range(h_bins):
        for s in range(s_bins):
            bin_val = cv.QueryHistValue_2D(hist, h, s)
            intensity = cv.Round(bin_val * 255 / max_value)
            cv.Rectangle(hist_img,
                         (h*scale, s*scale),
                         ((h+1)*scale - 1, (s+1)*scale - 1),
                         cv.RGB(intensity, intensity, intensity),
                         cv.CV_FILLED)
    return hist_img







def test_CalcEMD2():
    cc = {}
    for a,b,r in [ (10,10,5), (10,15,4) ]:
        scratch = cv.CreateImage((512,512), 8, 1)
        cv.SetZero(scratch)
        cv.Circle(scratch, (a,b), r, 255, -1)

        cv.NamedWindow("arr1", 1)
        cv.ShowImage("arr1", scratch)
        cv.WaitKey(0)

        storage = cv.CreateMemStorage()
        seq = cv.FindContours(scratch, storage, cv.CV_RETR_TREE, cv.CV_CHAIN_APPROX_SIMPLE)
        arr = cv.CreateMat(len(seq), 3, cv.CV_32FC1)
        for i,e in enumerate(seq):
            print "r ",r ,"i ",i, " e ",e
            arr[i,0] = 1
            arr[i,1] = e[0]
            arr[i,2] = e[1]
            cc[r] = arr

    def myL1(A, B, D):
        return abs(A[0]-B[0]) + abs(A[1]-B[1])
    def myL2(A, B, D):
        return math.sqrt((A[0]-B[0])**2 + (A[1]-B[1])**2)
    def myC(A, B, D):
        return max(abs(A[0]-B[0]), abs(A[1]-B[1]))

    contours = set(cc.values())
    for c0 in contours:
        for c1 in contours:
            print "##########"
            print cv.CalcEMD2(c0, c1, cv.CV_DIST_L1)
            print cv.CalcEMD2(c0, c1, cv.CV_DIST_L2)
            print cv.CalcEMD2(c0, c1, cv.CV_DIST_C)
            print cv.CalcEMD2(c0, c1, cv.CV_DIST_USER, myL1)
            print cv.CalcEMD2(c0, c1, cv.CV_DIST_USER, myL2)
            print cv.CalcEMD2(c0, c1, cv.CV_DIST_USER, myC)
            print "##########"
            #print "hhu ",abs(cv.CalcEMD2(c0, c1, cv.CV_DIST_L1) - cv.CalcEMD2(c0, c1, cv.CV_DIST_USER, myL1))
            #print "hoo ",abs(cv.CalcEMD2(c0, c1, cv.CV_DIST_L2) - cv.CalcEMD2(c0, c1, cv.CV_DIST_USER, myL2))
            #print "hid ",abs(cv.CalcEMD2(c0, c1, cv.CV_DIST_C) - cv.CalcEMD2(c0, c1, cv.CV_DIST_USER, myC))



def testEMD2():

    arr1 = cv.CreateMat(1,3,cv.CV_32FC1)
    arr1[0,0] = 0.1
    arr1[0,1] = 0.1
    arr1[0,2] = 0.8
    #arr1[1,1] = 0.1

    arr2 = cv.CreateMat(1,3,cv.CV_32FC1)
    arr2[0,0] = 0.2
    arr2[0,1] = 0.0
    arr2[0,2] = 0.8
    #arr2[1,1] = 0.9


    """

    arr1 = cv.CreateMat(100,100,cv.CV_32FC1)
    arr1[:,:] = 0.0001



    arr2 = cv.CreateMat(100,100,cv.CV_32FC1)
    arr2[:,:] = 0.0001

    for i in range(0,50):
        for j in range(0,100):
            arr1[i,j] = 1.0
    for i in range(50,100):
        for j in range(0,100):
            arr2[i,j] = 1.0
    """



    res = cv.CalcEMD2(arr1,arr2,cv.CV_DIST_C)

    print cv.CalcEMD2(arr1,arr2,cv.CV_DIST_C)," ", cv.CalcEMD2(arr1,arr2,cv.CV_DIST_L1)," ", cv.CalcEMD2(arr1,arr2,cv.CV_DIST_L2)

    """
    cv.NamedWindow("arr1", 1)
    cv.ShowImage("arr1", arr1)
    cv.NamedWindow("arr2", 1)
    cv.ShowImage("arr2", arr2)
    cv.WaitKey(0)
    """
    

    print res



def bruitFFT(imgGray):
    imfft = scipy.fftpack.fft2(imgGray)
    imfftm1 = imfft

    print imfftm1
    for i in range(imfft.shape[0]):
        for j in range(imfft.shape[1]):
            imfftm1[i,j] = numpy.complex128(imfft[i,j].real+imfft[i,j].imag)
            #imfftm1[i,j] = numpy.complex128(imfft[i,j]+imfft.imag[i,j])
            #imfftm1[i,j] = numpy.complex128(imfft[i,j*random.random()+imfft.imag[i,j])

    #imfftm1 = numpy.complex128(imfft.real[0,:]+imfft.imag[0,:])

    #print numpy.complex128(400+0j)

    
    

    imifft = numpy.real(scipy.fftpack.ifft2(imfftm1))
    

    return imifft




if __name__ == '__main__':

    src = cv.LoadImageM(sys.argv[1])
    cv.NamedWindow("Source", 1)
    cv.ShowImage("Source", src)

    cv.NamedWindow("H-S Histogram", 1)
    cv.ShowImage("H-S Histogram", hs_histogram(src))

    cv.WaitKey(0)
    """
    test_CalcEMD2()
    testEMD2()
    emd2FormShape()

    #imgGray = scipy.misc.imread('c:/kp03.bmp')
    #imgGray = 0.2989*imgGray[:,:,0]+ 0.5870*imgGray[:,:,1]+ 0.1140*imgGray[:,:,2]

    #imgR = bruitFFT(imgGray)

    #scipy.misc.imsave('c:/res.bmp',imgR)
    """