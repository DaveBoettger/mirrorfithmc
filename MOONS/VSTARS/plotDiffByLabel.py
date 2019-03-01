import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import mirrorfit.mirrorfit as mirrorfit
import sys

UPPER_LIM = 60
if __name__ == '__main__':
	ds1fName = sys.argv[1]
	ds2fName = sys.argv[2]
	try:
		title = ' '.join(sys.argv[3:])
	except:
		title = 'No Title'
	ds1 = mirrorfit.dataset()
	ds1.readDataFile(ds1fName)

	ds2 = mirrorfit.dataset()
	ds2.readDataFile(ds2fName)

	ds1_ind, ds2_ind = ds1.matchLabels(ds2)

	ds1a = ds1.getPoints(ds1_ind,copy=True)
	ds2a = ds2.getPoints(ds2_ind, copy=True)

	dx = ds1a.toArray() - ds2a.toArray()
	dx_err = np.sqrt(ds1a.toErrArray()**2 + ds2a.toErrArray()**2)


	dDiffsErr = np.sqrt(np.sum(dx_err**2,1))*1e3
	dDiffs = np.sqrt(np.sum(dx**2,1))*1e3
	print dDiffs/dDiffsErr
	
	plt.hist(dDiffs,30, range=[0,UPPER_LIM])
	if np.sum(dDiffs>UPPER_LIM):
		diffTitle = title+' ({0} elements of of bounds)'.format(np.sum(dDiffs>UPPER_LIM))
	else:
		diffTitle = title
	plt.title(diffTitle)
	plt.xlabel('Actual Distance Error [microns]')
	plt.ylabel('Number of Points')
	plt.figure()
	plt.hist(dDiffs/dDiffsErr, 30)
	plt.xlabel('Actual Distance Error / Error Estimate ($\sigma$ deviation)')
	plt.ylabel('Number of Points')
	plt.title(title)
	plt.figure()
	plt.plot(dDiffsErr, dDiffs/dDiffsErr, '.')
	plt.title(title)
	plt.xlabel('Error Estimate [microns]')
	plt.ylabel('Actual Distance Error / Error Estimate ($\sigma$ deviation)')
	plt.show()