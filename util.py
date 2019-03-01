import numpy as np
import scipy
import pymc3 as pm
import theano
import json

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

def generate_standard_transform_variables(fitmap):
        #We need a better way of defining the priors here, needs thought TODO
        #Translation variables:
        tvals = {}
        if fitmap['tx']:
            tvals['tx'] = pm.Normal('tx', mu=0, sd=10000.)
        else:
            tvals['tx'] = 0.
        if fitmap['ty']:
            tvals['ty'] = pm.Normal('ty', mu=0, sd=10000.)
        else:
            tvals['ty'] = 0.
        if fitmap['tz']:
            tvals['tz'] = pm.Normal('tz', mu=0, sd=10000.)
        else:
            tvals['tz'] = 0.
        #Rotation variables:
        if fitmap['rx']:
            tvals['rx'] = pm.Normal('rx',mu=0, sd=200)
        else:
            tvals['rx'] = 0.
        if fitmap['ry']:
            tvals['ry'] = pm.Normal('ry',mu=0, sd=200)
        else:
            tvals['ry'] = 0.
        if fitmap['rz']:
            tvals['rz'] = pm.Normal('rz',mu=0, sd=200)
        else:
            tvals['rz'] = 0.
        #Scale
        if fitmap['s']:
            tvals['s'] = pm.Normal('s', mu=100, sd=10.)
        else:
            tvals['s'] = 100.

        return tvals

def find_credible_levels(x,y,contour_targets=[.997,.954,.683]):
    '''Adapted from https://stackoverflow.com/questions/35225307/set-confidence-levels-in-seaborn-kdeplot'''

    # Make a 2d normed histogram
    H,xedges,yedges=np.histogram2d(x,y,bins=50,normed=True)

    norm=H.sum() # Find the norm of the sum

    # Set target levels as percentage of norm
    targets = [norm*contour for contour in contour_targets]

    # Take histogram bin membership as proportional to Likelihood
    # This is true when data comes from a Markovian process
    def objective(limit, target):
        w = np.where(H>limit)
        count = H[w]
        return count.sum() - target

    levels = []
    # Find levels by summing histogram to objective
    for target in targets:
        levels.append(scipy.optimize.bisect(objective, H.min(), H.max(), args=(target,)))

    # For nice contour shading with seaborn, define top level
    levels.insert(0,H.min())
    levels.append(H.max())
    return levels

def load_param_file(fName):

    if fName.endswith('.json'):
        with open(fName) as f:
            return json.load(f)

#The following is not used:
def find_op_dependencies(obj, val_list=None):
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
        return [v for v in val_list if type(v)==pm.model.FreeRV]

