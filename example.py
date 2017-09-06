#!/usr/bin/env python

from RasterSpace import DiamondSpace
from RasterSpace import Line
inst = DiamondSpace(1,2,3,4,5,6,7,8)
inst.addLines([Line(1,4,3,4)])
print(inst.calc_Vanp())
