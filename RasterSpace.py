"""RasterSpace library written on python."""
import cv2
import numpy as np


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
        self.pSpace = np.zeros((spaceSize, spaceSize), dtype=np.uint8)
        self.spaceSize = spaceSize
        self.Normalization = Normalization
        self.SubPixelRadius = SubPixelRadius
        self.height = height
        self.width = width
        self.searchRange = searchRange
        self.margin = margin
        self.vp = vp


    def getVisSpace(self, drawMax=True, pdd=2):
        maxVal = np.max(self.pSpace)

        if maxVal < 1:
            return np.zeros((self.spaceSize, self.spaceSize), dtype=np.uint8)

        visSpace = self.pSpace / float(maxVal) * 255

        if drawMax:
            raw_x, raw_y = self.find_maximum()
            x = int(round(raw_x))
            y = int(round(raw_y))
            visSpace = visSpace.astype(np.uint8)
            visSpace = cv2.cvtColor(visSpace, cv2.COLOR_GRAY2BGR)

            if self.vp == 2:
                # draw top search region
                substracted_data = self.spaceSize/2 - self.searchRange/2
                addicted_data = self.spaceSize/2 + self.searchRange/2
                cv2.rectangle(visSpace,
                              (int(substracted_data), self.margin),
                              (int(addicted_data), self.searchRange),
                              (0, 255, 0))

                # draw bottom search region
                cv2.rectangle(visSpace, (int(substracted_data), self.spaceSize - self.searchRange),
                                        (int(addicted_data), self.spaceSize - self.margin - 1), (0,255,0))

            # draw maximum point
            cv2.rectangle(visSpace, (x-pdd, y-pdd), (x+pdd, y+pdd), (0, 0, 255))
            return visSpace
        else:
            return visSpace.astype(np.uint8)

    def sgn(self, val):
        if val >= 0:
            return +1
        return -1

    def sign(self, val):
        if val > 0:
            return +1
        elif val == 0:
            return 0
        return -1

    def lines_end_points(self, lines, space_c):
        center = int(round(space_c))
        endpoints = []

        for line in lines:
            a = line[0]
            b = line[1]
            c = line[2]

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
        for line, endpoint in zip(lines, endpoints):
            weight = int(line[3])

            for i in range(0, 6, 2):
                x1 = endpoint[i]
                y1 = endpoint[i+1]
                x2 = endpoint[i+2]
                y2 = endpoint[i+3]
                data = (x1, y1, x2, y2, weight)
                if abs(y2 - y1) > abs(x2 - x1):
                    self.lineV(*data)
                else:
                    self.lineH(*data)

            self.pSpace[endpoint[7], endpoint[6]] += weight


    def addLines(self, lines):
        space_c = (self.spaceSize - 1.0)/2.0

        EndPoints = self.lines_end_points(lines, space_c)

        self.rasterize_lines(lines, EndPoints)

    def find_maximum(self):
        R = self.SubPixelRadius

        if self.vp == 1:
            dd = 5
            hs = int(self.spaceSize / 2)
            left = hs-dd
            right = hs+dd
            self.pSpace[(left):(right), (left):(right)] = 0

            y, x = np.unravel_index(self.pSpace.argmax(), self.pSpace.shape)
            y += 1
            x += 1
        else:
            substracted2_sizerange = int(self.spaceSize/2 - self.searchRange/2)
            addicted_sizerange = int(self.spaceSize/2 + self.searchRange/2)
            sub_sizerange = self.spaceSize - self.searchRange
            topRegion = self.pSpace[self.margin:self.searchRange,
                                    substracted2_sizerange:addicted_sizerange]

            bottomRegion = self.pSpace[(sub_sizerange):(self.spaceSize - self.margin),
                                       (substracted2_sizerange):(addicted_sizerange)]

            maxTop = np.max(topRegion)
            maxBottom = np.max(bottomRegion)

            if maxTop > maxBottom:
                y, x = np.unravel_index(topRegion.argmax(), topRegion.shape)
                y += 1
            else:
                y, x = np.unravel_index(bottomRegion.argmax(), bottomRegion.shape)
                y += (sub_sizerange) + 1

            x += substracted2_sizerange + 1

        oSize = 2 * self.SubPixelRadius + 1
        O = np.zeros((oSize, oSize), dtype=np.float)

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
                left = i+R
                right = j+R
                sumSR += O[left, right] * i
                sumSC += O[left, right] * j
                sumO += O[left, right]

        return (x + sumSC/sumO, y + sumSR/sumO)

    def normalize_PC_points(self, PC_VanP):
        return ((2 * PC_VanP[0] - (self.spaceSize + 1)) / (self.spaceSize - 1),
                (2 * PC_VanP[1] - (self.spaceSize + 1)) / (self.spaceSize - 1))


    def PC_point_to_CC(self, PC_NormVP):
        x = float(PC_NormVP[0])
        y = float(PC_NormVP[1])

        m = max(self.height, self.width)

        v1 = y / x
        w2 = (self.sign(y) * y + self.sign(x) * x - 1) / x
        u3 = 1.0

        return ((v1 / self.Normalization * (m - 1) + self.width + 1) / 2,
                (w2 / self.Normalization * (m - 1) + self.height + 1) / 2)


    def calc_Vanp(self):
        PC_VanP = self.find_maximum()

        PC_NormVP = self.normalize_PC_points(PC_VanP)

        CC_VanP = self.PC_point_to_CC(PC_NormVP)

        return CC_VanP
