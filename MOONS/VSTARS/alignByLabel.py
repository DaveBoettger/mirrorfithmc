import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import mirrorfit.mirrorfit as mirrorfit
import sys

if __name__ == '__main__':

	refFName = sys.argv[1]
	toAlignFName = sys.argv[2]

	toAlign = mirrorfit.dataset()
	toAlign.readDataFile(toAlignFName)

	ref = mirrorfit.dataset()
	ref.readDataFile(refFName)
	r_ind, a_ind = ref.matchLabels(toAlign)
	trans = toAlign.align(ref=ref, tmap=[1,1,1,1,1,1,1], my_ind=a_ind, ref_ind=r_ind, apply=False, iter=10)
	print trans
	quit()
	for l in state.alignmentSelectionList:
		use.append(l['ref'][0])
		ind.append(l['align'][0])
		trans = toAlign.align(ref=ref, tmap=[1,1,1,1,1,1,0], my_ind=ind, ref_ind=use, apply=False, iter=10)
