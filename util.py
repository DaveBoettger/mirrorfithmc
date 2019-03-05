import numpy as np
import scipy
import pymc3 as pm
import theano
import json
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

