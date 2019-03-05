try:
    import mirrorfithmc.mirrorfithmc as mf
except:
    print('except')
    import mirrorfithmc as mf

import numpy as np
import pymc3 as pm
import matplotlib.pyplot as plt

MEASURED_RECEIVER_TARGET_NAMES = ['PROJECTEDRECEIVERTARGETN1','PROJECTEDRECEIVERTARGETN2','PROJECTEDRECEIVERTARGETN3','PROJECTEDRECEIVERTARGETN4','PROJECTEDRECEIVERTARGETN5','PROJECTEDRECEIVERTARGETN6','PROJECTEDRECEIVERTARGETN7','RECEIVERTARGETT1','RECEIVERTARGETT2','RECEIVERTARGETT3','RECEIVERTARGETT4','RECEIVERTARGETT5','RECEIVERTARGETT6','RECEIVERTARGETT7','RECEIVERTARGETT8','RECEIVERTARGETT9']

THEORY_RECEIVER_TARGET_NAMES = ['FOCALPLANEPOINTTARGET', 'FOCALPLANEXDISPLACETARGET', 'FOCALPLANEYDISPLACETARGET', 'FOCALPLANEZDISPLACETARGET']

def add_receiver_theory_points(ds, output_file = None, theory_ds='POLARBEAR/pb2a_cryostat_japan_measurements.txt'):

    if type(ds) is str:
        ds = mf.Dataset(from_file=ds)
    elif type(ds) is mf.Dataset:
        ds = ds.copy()
    else:
        raise ValueError('Type of dataset argument not recoginized.')

    if type(theory_ds) is str:
        theoryds = mf.Dataset(from_file=theory_ds)
    elif type(theory_ds) is mf.Dataset:
        theoryds = theory_ds.copy()
    else:
        raise ValueError('Type of theory dataset argument not recoginized.')

    dsuse = ds.subset_from_labels(MEASURED_RECEIVER_TARGET_NAMES)

    with mf.AlignDatasets(ds1=dsuse, ds2=theoryds, fitmap={'s':False}) as tamodel:
        tatrace = pm.sample(2000, tune=5500, init = 'advi+adapt_diag', nuts_kwargs={'target_accept': .90, 'max_treedepth': 25}, error_scale1=1., error_scale2=1.) 
    pm.save_trace(tatrace)
    pm.traceplot(tatrace)
    plt.show()
    fptheory = theoryds.subset_from_marker('FOCALPLANE')
    pos, err = tamodel.use_transform_trace(fptheory.to_tensors(), tatrace)
    newtheoryarray=mf.DatasetArrays(pos=np.mean(pos,axis=0), err=np.std(pos,axis=0), serr=np.std(pos,axis=0))
    newfptheory=fptheory.remake_from_arrays(newtheoryarray)
    for p in newfptheory.values():
        ds.add_point(p)
    print(ds)
    
if __name__ == '__main__':
    add_receiver_theory_points(ds='20181208_primary_receiver.txt')

