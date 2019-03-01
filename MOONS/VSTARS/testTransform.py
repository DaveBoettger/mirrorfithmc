import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import mirrorfit.mirrorfit as mirrorfit
import sys

#transform = ([0.024717367271067409, -0.008842983526859971, 0.040511573439941911, -0.00015556255915115828, -7.9612473586916153e-05, -8.2451918351369411e-06, 1.00009986456], [1,1,1,1,1,1,1]) 1->2
transform = ([0.0020882876672506778, 0.0092900387078673659, -0.016596853624808117, 8.0578471698683937e-05, 1.018592834686056e-06, 2.0746370713169883e-05, 0.999953288547], [1,1,1,1,1,1,1]) # 3->2

identity = ([0., 0., 0., 0., 0., 0., 1.], [1,1,1,1,1,1,1])


if __name__ == '__main__':
	
	refFName = sys.argv[1]
	toAlignFName = sys.argv[2]
	outputFName = sys.argv[3]

	toAlign = mirrorfit.dataset()
	toAlign.readDataFile(toAlignFName)

	ref = mirrorfit.dataset()
	ref.readDataFile(refFName)

	r_ind, a_ind = ref.matchLabels(toAlign)

	trans = mirrorfit.transform(*transform)
	ident = mirrorfit.transform(*identity)
	aref = ref.getPoints(r_ind,copy=True)
	ds = toAlign.getPoints(a_ind, copy=True)

	print ds.chi2(ident.t, ident.tmap, aref)

	trans*ds

	print ds.chi2(ident.t, ident.tmap, aref)

	ds.export(outputFName)