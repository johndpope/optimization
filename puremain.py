import math
import cv2
import numpy as np
import time
import sys
import math
import cv2
import numpy as np
import time
import sys




class Line(object):

    def __init__(self, a, b, c, d):
        self.a = float(a)
        self.b = float(b)
        self.c = float(c)
        self.d = float(d)

class DiamondSpace:

    def __init__(self, spaceSize, height, width, searchRange, Normalization, SubPixelRadius, margin, vp):
        self.pSpace = np.zeros((spaceSize, spaceSize), dtype = np.uint)
        self.spaceSize = spaceSize
        self.Normalization = Normalization
        self.SubPixelRadius = SubPixelRadius
        self.height = height
        self.width = width
        self.searchRange = searchRange
        self.margin = margin
        self.vp = vp


    def getVisSpace(self, drawMax = True, pdd = 2):
        maxVal = np.max(self.pSpace)

        if maxVal < 1:
            return np.zeros((self.spaceSize, self.spaceSize), dtype = np.uint8)

        visSpace = self.pSpace / float(maxVal) * 255

        if drawMax:
            x, y = self.find_maximum()
            x = int(round(x))
            y = int(round(y))
            visSpace = visSpace.astype(np.uint8)
            visSpace = cv2.cvtColor(visSpace, cv2.COLOR_GRAY2BGR)
            
            if self.vp == 2:
                # draw top search region
                cv2.rectangle(visSpace, (int(self.spaceSize/2 - self.searchRange/2), self.margin), 
                                        (int(self.spaceSize/2 + self.searchRange/2), self.searchRange), (0,255,0))

                #draw bottom search region
                cv2.rectangle(visSpace, (int(self.spaceSize/2 - self.searchRange/2), self.spaceSize - self.searchRange), 
                                        (int(self.spaceSize/2 + self.searchRange/2), self.spaceSize - self.margin - 1), (0,255,0))

            # draw maximum point
            cv2.rectangle(visSpace, (x-pdd, y-pdd), (x+pdd, y+pdd), (0,0,255))
            return visSpace
        else:
            return visSpace.astype(np.uint8)


    def sgn(self, val):
        return (0 <= val) - (val < 0)


    def sign(self, val):
        return (0 <= val) - (val <= 0)        


    def lines_end_points(self, lines, space_c):
        center = int(round(space_c))
        endpoints = []

        for line in lines:
            a = line.a
            b = line.b
            c = line.d

            alpha = float(self.sgn(a*b))
            beta = float(self.sgn(b*c))
            gamma = float(self.sgn(a*c))
            
            a_x = alpha*a / (c + gamma*a)
            b_x = -alpha*c / (c + gamma*a)

            end1 = int(round((a_x + 1) * space_c))
            end0 = int(round((b_x + 1) * space_c))

            end3 = int(round((b / (c + beta * b) + 1) * space_c))
            end2 = center

            end5 = center
            end4 = int(round((b / (a + alpha * b) + 1) * space_c))

            end7 = int(round((-a_x + 1) * space_c))
            end6 = int(round((-b_x + 1) * space_c))

            endpoints.append((end0, end1, end2, end3, end4, end5, end6, end7))

        return endpoints



    def lineV(self, x0, y0, x1, y1, weight):
        slope = (x1 - x0) / float(y1 - y0)

        x_start = float(x0) + 0.5
        x_iter = x_start
        step = 1 if y0 < y1 else -1
        slope *= step

        y = y0
        c = 1

        while y != y1:
            self.pSpace[int(x_iter), y] += weight
            x_iter = x_start + c * slope
            y += step
            c += 1



    def lineH(self, x0, y0, x1, y1, weight):
        slope = (y1 - y0) / float(x1 - x0 - 0.00001)

        y_start = float(y0) + 0.5
        y_iter = y_start
        step = 1 if x0 < x1 else -1
        slope *= step

        x = x0
        c = 1

        while x != x1:
            self.pSpace[x, int(y_iter)] += weight
            y_iter = y_start + c * slope
            x += step
            c += 1



    def rasterize_lines(self, lines, endpoints):
        for line, end in zip(lines, endpoints):
            weight = int(line.d)            

            for i in range(0,6,2):
                if abs(end[i+3] - end[i+1]) > abs(end[i+2] - end[i]):
                    self.lineV(end[i], end[i+1], end[i+2], end[i+3], weight)
                else:
                    self.lineH(end[i], end[i+1], end[i+2], end[i+3], weight)

            self.pSpace[end[7],end[6]] += weight


    def addLines(self, lines):
        space_c = (self.spaceSize - 1.0)/2.0

        EndPoints = self.lines_end_points(lines, space_c)

        self.rasterize_lines(lines, EndPoints)



    def find_maximum(self):
        R = self.SubPixelRadius

        if self.vp == 1:
            dd = 5
            hs = int(self.spaceSize / 2)
            self.pSpace[(hs-dd):(hs+dd),(hs-dd):(hs+dd)] = 0

            y, x = np.unravel_index(self.pSpace.argmax(), self.pSpace.shape)
            y += 1
            x += 1
        else:
            topRegion = self.pSpace[self.margin:self.searchRange, 
                                    int(self.spaceSize/2 - self.searchRange/2):int(self.spaceSize/2 + self.searchRange/2)]

            bottomRegion = self.pSpace[(self.spaceSize - self.searchRange):(self.spaceSize - self.margin),
                                       (int(self.spaceSize/2 - self.searchRange/2)):(int(self.spaceSize/2 + self.searchRange/2))]

            maxTop = np.max(topRegion)
            maxBottom = np.max(bottomRegion)

            if maxTop > maxBottom:
                y, x = np.unravel_index(topRegion.argmax(), topRegion.shape)
                y += 1
            else:
                y, x = np.unravel_index(bottomRegion.argmax(), bottomRegion.shape)
                y += (self.spaceSize - self.searchRange) + 1

            x += int(self.spaceSize/2 - self.searchRange/2) + 1



        oSize = 2 * self.SubPixelRadius + 1
        O = np.zeros((oSize, oSize), dtype = np.float)

        ist = y - R
        iend = y + R + 1

        jst = x - R
        jend = x + R + 1

        for i in range(ist, iend):
            for j in range(jst, jend):
                if i > 0 and i < self.spaceSize and j > 0 and j < self.spaceSize:
                    O[i - ist, j - jst] = self.pSpace[i, j]


        sumSR = 0.0
        sumSC = 0.0
        sumO = 0.0

        for i in range(-R, R+1):
            for j in range(-R, R+1):
                sumSR += O[i+R, j+R] * i
                sumSC += O[i+R, j+R] * j
                sumO += O[i+R, j+R]

        return x + sumSC/sumO, y + sumSR/sumO


    def normalize_PC_points(self, PC_VanP):
        return (2 * PC_VanP[0] - (self.spaceSize + 1)) / (self.spaceSize - 1), (2 * PC_VanP[1] - (self.spaceSize + 1)) / (self.spaceSize - 1)


    def PC_point_to_CC(self, PC_NormVP):
        x = float(PC_NormVP[0])
        y = float(PC_NormVP[1])

        m = max(self.height, self.width)

        v1 = y / x
        w2 = (self.sign(y) * y + self.sign(x) * x - 1) / x
        u3 = 1.0

        return (v1 / self.Normalization * (m - 1) + self.width + 1) / 2, (w2 / self.Normalization * (m - 1) + self.height + 1) / 2


    def calc_Vanp(self):
        PC_VanP = self.find_maximum()
        # print PC_VanP, "\n"

        PC_NormVP = self.normalize_PC_points(PC_VanP)
        # print PC_NormVP, "\n"

        CC_VanP = self.PC_point_to_CC(PC_NormVP)

        return CC_VanP



videoPath = "./challenge_video.mp4"

# cap = cv2.VideoCapture(0)
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

        # ret, frame = cap.read()

        ret, frame = cap.read()
        img = cv2.flip(frame, 1)
        cv2.imshow('my webcam', frame)
        if cv2.waitKey(1) == 27: 
            break  # esc to quit
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
                print x, y

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
