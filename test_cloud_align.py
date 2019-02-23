import mirrorfit as mf
import pymc3 as pm
import numpy as np
import matplotlib.pyplot as plt
import scipy

def load_model():
    ds1 = mf.dataset(from_file='20181205_primary_receiver.txt')
    ds2 = mf.dataset(from_file='20181208_primary_receiver.txt') 
    with mf.AlignDatasetSimple(ds1=ds1,ds2=ds2, use_marker='PRIMARY', fitmap={'tx':True, 'ty':True, 'tz':True, 's':True, 'rx':True, 'ry':True, 'rz':True}) as model: 
        return model

def find_map(model):
    with model as model: 
        return pm.find_MAP(model=model)

def sample(model):

    with model as model: 
        trace = pm.sample(2000, tune=10000, init = 'advi+adapt_diag', nuts_kwargs={'target_accept': .95, 'max_treedepth': 25}) 

    return trace

if __name__ == '__main__':

    model = load_model()
    print(find_map(model))
    trace = sample(model)
    pm.save_trace(trace)
