import mirrorfit as mf
import pymc3 as pm
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':

    ds1 = mf.dataset(from_file='20181205_primary_receiver.txt')
    ds2 = mf.dataset(from_file='20181208_primary_receiver.txt') 
    with mf.AlignDatasetSimple(ds1=ds1,ds2=ds2) as model: 
        trace = pm.sample(2000, tune=1000, init = 'advi+adapt_diag', nuts_kwargs={'target_accept': .95, 'max_treedepth': 15}) 
     
    distances = trace.get_values('Alignment_distance') 
    errors = trace.get_values('Alignment_errors')
