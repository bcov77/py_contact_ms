#!/usr/bin/env python

import os
import sys
import numpy as np

from pyrosetta import *
from pyrosetta.rosetta import *

init('-out:levels core.scoring.sc:500')

sys.path.append('/home/bcov/sc/random/npose')
import npose_util as nu


pose = pose_from_file(sys.argv[1])


calc = core.scoring.sc.MolecularSurfaceCalculator()
calc.Calc(pose, 1)



dots0 = []
for dot in calc.GetDots(0):
    dots0.append(np.array([dot.coor(1), dot.coor(2), dot.coor(3)]))

dots0 = np.array(dots0)

dots1 = []
for dot in calc.GetDots(1):
    dots1.append(np.array([dot.coor(1), dot.coor(2), dot.coor(3)]))
dots1 = np.array(dots1)


