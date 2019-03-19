try:
    import mirrorfithmc.mirrorfithmc as mf
except:
    import mirrorfithmc as mf
import pymc3 as pm
import numpy as np
import matplotlib.pyplot as plt
import scipy

def load_model_primary():
    ds1 = mf.Dataset(from_file='20181205_primary_receiver.txt', name='ds20181205')
    ds2 = mf.Dataset(from_file='20181208_primary_receiver.txt', name='ds20181208') 
    with mf.AlignDatasets(ds1=ds1,ds2=ds2, use_marker='PRIMARY', fitmap={'tx':True, 'ty':True, 'tz':True, 's':True, 'rx':True, 'ry':True, 'rz':True, 'rescale_errors':True}) as model: 
        return model

def load_multi_model_primary():
    ds1 = mf.Dataset(from_file='20181205_primary_receiver.txt', name='ds20181205')
    ds2 = mf.Dataset(from_file='20181208_primary_receiver.txt', name='ds20181208') 
    ds3 = mf.Dataset(from_file='../PB_point_clouds/SA_NORTH_FIELD/primary_driver.txt', name='PRIMARY_DRIVER')
    ds4 = mf.Dataset(from_file='../PB_point_clouds/SA_NORTH_FIELD/20171214_2/driveraligned_2.txt', name='ds20171214')

    with mf.AlignManyDatasets(reference=ds1, datasets=[ds2,ds3,ds4], use_marker='PRIMARY') as model:
        return model

def load_model_add_reciver_points():
    ds1 = mf.Dataset(from_file='20181208_primary_receiver.txt', name='ds20181208') 
    theory_set = mf.Dataset(from_file='')

def load_model_align_primary():
    ds1 = mf.Dataset(from_file='20181205_primary_receiver.txt', name='ds20181205')
    ds2 = mf.Dataset(from_file='20181208_primary_receiver.txt', name='ds20181208') 
    ds3 = mf.Dataset(from_file='/home/dave/dev/PB/Photogrammetry_20181119/Photogrammetry/PB_point_clouds/COSPOL/20190319/REFLECTORAcheckout.txt')
    #ds3 = mf.Dataset(from_file='../PB_point_clouds/SA_NORTH_FIELD/primary_driver.txt', name='PRIMARY_DRIVER')
    with mf.AlignMirror2(ds=ds1, mirror_definition = './POLARBEAR/SA_Primary_North.json', use_marker='PRIMARY', fitmap={'tx':True, 'ty':True, 'tz':True, 'rx':True, 'ry':True, 'rz':False, 's':False, 'R':True, 'mirror_std':True }) as model: 
        return model

def load_model_moons():
    ds1 = mf.Dataset(from_file='./MOONS/VSTARS/moons_20160802_1_aligned.txt', name='DS1')
    ds2 = mf.Dataset(from_file='./MOONS/VSTARS/moons_20160802_2_aligned.txt', name='DS2') 
    ds3 = mf.Dataset(from_file='./MOONS/VSTARS/moons_20160802_3_aligned.txt', name='DS3') 
    with mf.AlignDatasets(ds1=ds1,ds2=ds2, use_marker='TARGET', fitmap={'tx':True, 'ty':True, 'tz':True, 's':True, 'rx':True, 'ry':True, 'rz':True}) as model: 
        return model

def load_multi_model_moons():
    ds1 = mf.Dataset(from_file='./MOONS/VSTARS/moons_20160802_1_aligned.txt', name='DS1')
    ds2 = mf.Dataset(from_file='./MOONS/VSTARS/moons_20160802_2_aligned.txt', name='DS2') 
    ds3 = mf.Dataset(from_file='./MOONS/VSTARS/moons_20160802_3_aligned.txt', name='DS3') 
    with mf.AlignManyDatasets(reference=ds1, datasets=[ds2,ds3], use_marker='TARGET') as model:
        return model

def find_map(model):
    with model as model: 
        return pm.find_MAP(model=model)

def sample(model):

    with model as model: 
        trace = pm.sample(2000, tune=800, init = 'advi+adapt_diag', nuts_kwargs={'target_accept': .90, 'max_treedepth': 25}) 
        #trace = pm.sample(2000, tune=5500, init = 'jitter+adapt_diag', nuts_kwargs={'target_accept': .90, 'max_treedepth': 25}) 

        return trace

if __name__ == '__main__':

    #model = load_multi_model_moons()
    model = load_model_align_primary()
    #model = load_model_primary()
    #model = load_multi_model_primary()
    print(model.vars, model.test_point)
    trace = sample(model)
    pm.save_trace(trace)
    pm.traceplot(trace)
    plt.show()
