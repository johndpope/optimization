import math
import cv2
import numpy as np
import time
import sys
from RasterSpace import DiamondSpace
from RasterSpace import Line

videoPath = "./226.mp4"

cap = cv2.VideoCapture(videoPath)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
print "Width", width
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print "Height", height

# cap.set(cv2.CAP_PROP_POS_FRAMES, 2100)

print "starting ..."

# init variables

framenum = 0
scale = 0.5
scaleSpaceSize = 0.5;

houghTransfromThreshold = 90;
minAngle = np.pi/3.0
maxAngle = np.pi/1.8

######### init backgroundSubstractorMOG2
backsub_params = dict(  history = 100,
                        varThreshold= 7,
                        detectShadows=False )

backsub = cv2.createBackgroundSubtractorMOG2(**backsub_params)
area = width * height * scale * scale;
mask_thr = area * 0.06;

kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7), (-1, -1))
kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3), (-1, -1))
kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3), (-1, -1))

######### init variables for normalization
w_c = (width * scale - 1)/2.0
h_c = (height * scale - 1)/2.0
norm = (max(w_c, h_c) - 14)

######### initialize DiamondSpace
diamondSpace_params = dict( Normalization = 1,
                            SubPixelRadius = 2,
                            margin = 0,
                            vp = 2)

SpaceSize = int(height * scaleSpaceSize);
searchRange = SpaceSize / 20;

diamondSpace = DiamondSpace(SpaceSize, int(height * scale), int(width * scale), searchRange, **diamondSpace_params)

print "Space Size: ", SpaceSize

sum_time = 0
sum_lines = 0

######### Run main loop
while (cap.isOpened()):

        ret, frame = cap.read()

        if ret:

            frame_rz = cv2.resize(frame, (0,0), fx = scale, fy = scale)
            mask = backsub.apply(frame_rz, None, 0.01)

            if framenum > 30 and np.count_nonzero(mask) > mask_thr:

                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
                mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel_dilate)
                maskBigger = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel_dilate)

                output = cv2.bitwise_and(frame_rz, frame_rz, mask = maskBigger)
                canny = cv2.Canny(output, 100, 200)
                edges = cv2.bitwise_and(canny, canny, mask = mask)

                cdst = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

                lines = cv2.HoughLines(edges, 1, np.pi/180, houghTransfromThreshold, min_theta =  minAngle, max_theta = maxAngle)

                if lines is None:
                    continue

                lines = np.reshape(lines,(lines.shape[0],2))
                sum_lines += len(lines)
                mxlines = []

                for rho,theta in lines:
                    a = np.cos(theta)
                    b = np.sin(theta)
                    x0 = a*rho
                    y0 = b*rho
                    x1 = int(x0 + 1000*(-b))
                    y1 = int(y0 + 1000*(a))
                    x2 = int(x0 - 1000*(-b))
                    y2 = int(y0 - 1000*(a))

                    cv2.line(cdst,(x1,y1),(x2,y2),(0,0,255),1)

                    # normalize lines
                    aa = y2 - y1
                    if aa==0:
                        continue

                    bb = x1 - x2
                    t = math.sqrt(aa * aa + bb * bb)
                    aa = aa / t
                    bb = bb / t
                    c = 0 -(y1 - h_c) * bb - (x1 - w_c) * aa

                    # mxlines.append((aa, bb, c/norm, 1))
                    mxlines.append(Line(aa, bb, c/norm, 1))


                st = time.time()
                diamondSpace.addLines(mxlines)
                sum_time += time.time() - st


                x, y = diamondSpace.calc_Vanp()
                print x,y,"\n"

                # cv2.imshow('frame', cv2.resize(frame, (0,0), fx=0.5, fy=0.5))
                cv2.imshow('Output', output)
                cv2.imshow('Canny', cdst)
                cv2.imshow('VP2 accum', diamondSpace.getVisSpace())

                k = cv2.waitKey(1)
                if k == 27:
                    break

            framenum += 1

        else:
            break

# 0.0234
print "average time:", sum_time/ (framenum - 30)
print "average time per line:", sum_time/ sum_lines

cap.release()
cv2.destroyAllWindows()
