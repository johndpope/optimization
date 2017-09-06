"""Python wrapper for RasterSpace."""

import RasterSpaceLib
import numpy as np

class Line(object):
    def __init__(self, a, b, c, d):
        self.a = float(a)
        self.b = float(b)
        self.c = float(c)
        self.d = float(d)

class DiamondSpace(object):

    def __init__(self,
                 spaceSize,
                 height,
                 width,
                 searchRange,
                 Normalization,
                 SubPixelRadius,
                 margin,
                 vp):
        self.pSpace = [[0 for i in range(spaceSize)] for j in range(spaceSize)]
        self.spaceSize = spaceSize
        self.Normalization = Normalization
        self.SubPixelRadius = SubPixelRadius
        self.height = height
        self.width = width
        self.searchRange = searchRange
        self.margin = margin
        self.vp = vp

    def addLines(self, lines):
        RasterSpaceLib.addLines(self.pSpace, lines, self.spaceSize)

    def calc_Vanp(self):
        return RasterSpaceLib.calc_CC_Vanp(self.pSpace,
                                           self.spaceSize,
                                           self.Normalization,
                                           self.height,
                                           self.width,
                                           self.SubPixelRadius,
                                           self.searchRange,
                                           self.margin,
                                           self.vp)


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
