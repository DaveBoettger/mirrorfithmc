import mirrorfithmc.mirrorfithmc as mf
import pymc3 as pm
import matplotlib.pyplot as plt

MEASURED_RECEIVER_TARGET_NAMES = ['PROJECTEDRECEIVERTARGETN1','PROJECTEDRECEIVERTARGETN2','PROJECTEDRECEIVERTARGETN3','PROJECTEDRECEIVERTARGETN4','PROJECTEDRECEIVERTARGETN5','PROJECTEDRECEIVERTARGETN6','PROJECTEDRECEIVERTARGETN7','RECEIVERTARGETT1','RECEIVERTARGETT2','RECEIVERTARGETT3','RECEIVERTARGETT4','RECEIVERTARGETT5','RECEIVERTARGETT6','RECEIVERTARGETT7','RECEIVERTARGETT8','RECEIVERTARGETT9']

def add_receiver_theory_points(ds, theory_ds='POLARBEAR/pb2a_cryostat_japan_measurements.txt'):

    ds = mf.Dataset(from_file=ds, name='MEASURED')
    ds = ds.subset_from_labels(MEASURED_RECEIVER_TARGET_NAMES)
    theoryds = mf.Dataset(from_file=theory_ds, name='THEORY')
    ds, theoryds = ds.subsets_in_common(theoryds)
    with mf.AlignDatasets(ds1=ds, ds2=theoryds, fitmap={'s':False}) as model:
        trace = pm.sample(2000, tune=5500, init = 'advi+adapt_diag', nuts_kwargs={'target_accept': .90, 'max_treedepth': 25}) 
    pm.traceplot(trace)
if __name__ == '__main__':
    add_receiver_theory_points(ds='20181208_primary_receiver.txt')

