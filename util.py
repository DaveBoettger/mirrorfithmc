import numpy as np
import pymc3 as pm
import theano
import json
import theano.tensor as tt
try:
    import mirrorfithmc.lib_transform as lt
except:
    import lib_transform as lt

#Use these to provide a fairly weak prior on variables used to scale measured errors
global ERROR_SCALE_BOUND
ERROR_SCALE_BOUND = pm.Bound(pm.StudentT,lower=0.0)
global ERROR_SCALE_NU 
ERROR_SCALE_NU = 1

def generate_error_scale_distribution(name, nu=ERROR_SCALE_NU, mu=1, sd=1, shape='xyz'):
    #This breaks for now if shape != 3, but should find a way to make this work for a series of different shapes TODO
    if shape=='xyz':
        return ERROR_SCALE_BOUND(name, nu=nu, mu=mu, sd=sd, shape=3)[:,np.newaxis]
    elif shape=='scalar':
        return ERROR_SCALE_BOUND(name, nu=nu, mu=mu, sd=sd, shape=1)
    else:
        raise NotImplementedError(f'shape argument {shape} is not supported.')

def generate_alignment_distribution(name, sd, observed, nu=np.inf):
    '''Returns alignment probabilities given observations and errors 
    This is used to centralize defaults.     

    If nu is infinite or None, then a Normal distribution is used.
    If nu is any other value, a StudenT_{nu} distribution is used.
    '''
    if nu is np.inf or nu is None:
        align = pm.Normal(name, mu=0, sd=sd, observed=observed)
    else:
        align = pm.StudentT(name, nu=nu, mu=0, sd=sd, observed=observed)

    return align

def generate_standard_transform_variables(fitmap=None, mus=None, sds=None):
        #We need a better way of defining the priors here, needs thought TODO
        #Translation variables:
        translate_keys = ['tx', 'ty', 'tz']
        #Rotation variables:
        rotate_keys = ['rx', 'ry', 'rz']
        #Scale variable:
        scale_keys = ['s']

        transfactor = lt.TheanoTransform.translate_factor
        fullscale = lt.TheanoTransform.full_scale
        rotatescale = lt.TheanoTransform.rotation_scale
        default_fitmap = {'tx':False,'ty':False,'tz':False,'rx':False,'ry':False,'rz':False,'s':False}
        default_means = {'tx':0., 'ty':0.,'tz':0.,'rx':0.,'ry':0., 'rz':0., 's':1.}
        default_stds = {'tx':10,'ty':10,'tz':10,'rx':np.pi,'ry':np.pi, 'rz':np.pi, 's':.1}

        if fitmap is not None:
            default_fitmap.update(fitmap)
        if mus is not None:
            default_means.update(mus)
        if sds is not None:
            default_stds.update(sds)

        tvals = {}
        for k in translate_keys:
            if default_fitmap[k]:
                tvals[k] = pm.Normal(k, mu=default_means[k]*transfactor, sd=default_stds[k]*transfactor)
            else: 
                tvals[k] = default_means[k]*transfactor
        for k in rotate_keys:
            if default_fitmap[k]:
                tvals[k] = pm.Normal(k, mu=default_means[k]*rotatescale, sd=default_stds[k]*rotatescale)
            else: 
                tvals[k] = default_means[k]*rotatescale
        for k in scale_keys:
            if default_fitmap[k]:
                tvals[k] = pm.Normal(k, mu=default_means[k]*fullscale, sd=default_stds[k]*fullscale)
            else: 
                tvals[k] = default_means[k]*fullscale
        return tvals

def update_fitmap(fitmap, default_fitmap):
    if fitmap is not None:
        for k in fitmap:
            if k not in default_fitmap:
                print(f'Warning: item {k} appears in fitmap but is not used for this alignment.')
            default_fitmap.update(fitmap)

    return default_fitmap

def load_param_file(fName):

    if fName.endswith('.json'):
        with open(fName) as f:
            return json.load(f)

def find_op_dependencies(obj, val_list=None):
    '''This can be useful for debugging'''
    if val_list is None:
        #top level call
        top_level = True
        val_list = []
    else:
        top_level = False
    parents = obj.get_parents()
    if len(parents):
        [find_op_dependencies(p,val_list) for p in parents]
    else:
        val_list.append(obj)
    if top_level:
        val_list = list(set(val_list))
        freervs =  [v for v in val_list if type(v)==pm.model.FreeRV]
        constants = [v for v in val_list if type(v)==tt.TensorConstant]
        return {'freervs':freervs,'constants':constants}

def trace_iterator(val, model, trace):
    '''Returns an iterator over val that calculates val for model at each point in trace
    val must be a Theano variable or a list of Theano variables.
    '''
    inputs = list(model.vars)
    inputs.extend(model.deterministics)
    this_fun=theano.function(inputs=inputs, outputs=val, on_unused_input='ignore')
    for p in trace.points():
        yield this_fun(**p)

def dict_trace_iterator(dictionary, model, trace):
    '''
    Returns an iterator over the values in dictionary, calculating the values for model at each point in trace
    values in the dictionary must be Theano variables.
    '''
    tlist = trace_iterator(val=list(dictionary.values()), model=model, trace=trace)
    for t in tlist:
        yield dict(zip(dictionary.keys(),t))

def trace_array(val, model, trace):
    this_it = trace_iterator(val, model, trace)
    ret_list = []
    for t in this_it:
        ret_list.append(t)
    
    return np.array(ret_list)

def trace_dict(dictionary, model, trace):
    this_it = dict_trace_iterator(dictionary, model, trace)
    this_dict={}
    for k in dictionary:
        this_dict[k]=[]
    for i in this_it:
        for k in dictionary:
            this_dict[k].append(i[k])
    for k in this_dict:
        this_dict[k] = np.array(this_dict[k])
    return this_dict

#def recover_dict_trace(dictionary, model, trace):
#    inputs = list(model.vars)
#    inputs.extend(model.deterministics)
#    this_fun=theano.function(inputs=inputs, outputs=list(dictionary.values()), on_unused_input='ignore')
#    for p in trace.points():
#        yield dict(zip(dictionary.keys(),this_fun(**p)))
