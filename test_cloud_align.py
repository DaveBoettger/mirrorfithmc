try:
    import mirrorfithmc as mf
    mf.Dataset
except:
    import mirrorfithmc.mirrorfithmc as mf
import pymc3 as pm
import numpy as np
import matplotlib.pyplot as plt
import scipy

def load_model_primary():
    ds1 = mf.dataset(from_file='20181205_primary_receiver.txt')
    ds2 = mf.dataset(from_file='20181208_primary_receiver.txt') 
    with mf.AlignDatasetSimple(ds1=ds1,ds2=ds2, use_marker='PRIMARY', fitmap={'tx':True, 'ty':True, 'tz':True, 's':True, 'rx':True, 'ry':True, 'rz':True}) as model: 
        return model

def load_model_moons():
    ds1 = mf.Dataset(from_file='/Users/daveboettger/Google Drive/PbGeneral/Photogrammetry/mirrorfithmc/MOONS/VSTARS/moons_20160802_1_aligned.txt', name='DS1')
    ds2 = mf.Dataset(from_file='/Users/daveboettger/Google Drive/PbGeneral/Photogrammetry/mirrorfithmc/MOONS/VSTARS/moons_20160802_2_aligned.txt', name='DS2') 
    ds3 = mf.Dataset(from_file='/Users/daveboettger/Google Drive/PbGeneral/Photogrammetry/mirrorfithmc/MOONS/VSTARS/moons_20160802_3_aligned.txt', name='DS3') 
    with mf.AlignDatasets(ds1=ds1,ds2=ds2, use_marker='TARGET', fitmap={'tx':True, 'ty':True, 'tz':True, 's':True, 'rx':True, 'ry':True, 'rz':True}) as model: 
        return model

def load_multi_model_moons():
    ds1 = mf.Dataset(from_file='/Users/daveboettger/Google Drive/PbGeneral/Photogrammetry/mirrorfithmc/MOONS/VSTARS/moons_20160802_1_aligned.txt', name='DS1')
    ds2 = mf.Dataset(from_file='/Users/daveboettger/Google Drive/PbGeneral/Photogrammetry/mirrorfithmc/MOONS/VSTARS/moons_20160802_2_aligned.txt', name='DS2') 
    ds3 = mf.Dataset(from_file='/Users/daveboettger/Google Drive/PbGeneral/Photogrammetry/mirrorfithmc/MOONS/VSTARS/moons_20160802_3_aligned.txt', name='DS3') 
    with mf.AlignManyDatasets(reference=ds1, datasets=[ds2,ds3], use_marker='TARGET') as model:
        return model

def find_map(model):
    with model as model: 
        return pm.find_MAP(model=model)

def sample(model):

    with model as model: 
        trace = pm.sample(2000, tune=15500, init = 'advi+adapt_diag', nuts_kwargs={'target_accept': .99, 'max_treedepth': 25}) 

    return trace

if __name__ == '__main__':

    model = load_multi_model_moons()
    print(model.vars, model.test_point)
    trace = sample(model)
    pm.save_trace(trace)
