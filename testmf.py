import pymc3 as pm
import numpy as np
import matplotlib.pyplot as plt
import theano.tensor as tt
import theano
import mirrorfit as mf
import json
import time

def dist(val1, val2):

    return tt.sqrt(tt.pow(val1,2)+tt.pow(val2,2))

def matrix2euler(R):
    '''Return equivalent rotations (in radians) around x, y, and z axes given a rotation matrix'''
    oz = np.arctan2(R[1][0],R[0][0])
    oy = np.arctan2(-R[2][0],np.sqrt(R[2][1]**2 + R[2][2]**2))
    ox = np.arctan2(R[2][1],R[2][2])
    return (ox, oy, oz)

def rot(do):
    '''Build a rotation matrix given a tuple of three angles (in radians) for rotations around x, y, and z.'''
    ox,oy,oz = do
    Rx = np.array([[1,0,0],[0,np.cos(ox),-np.sin(ox)],[0,np.sin(ox),np.cos(ox)]])
    Ry = np.array([[np.cos(oy),0,np.sin(oy)],[0,1,0],[-np.sin(oy),0,np.cos(oy)]])
    Rz = np.array([[np.cos(oz),-np.sin(oz),0],[np.sin(oz),np.cos(oz),0],[0,0,1]])
    R = np.dot(Rz,np.dot(Ry,Rx))
    return R

def theano_rot(rx,ry,rz, rescale=True):
    '''Return a theano tensor representing a rotation 
    matrix using specified rotation angles rx,ry, rz'''

    if rescale:
        rx = np.pi/180. * (rx)
        ry = np.pi/180. * (ry)
        rz = np.pi/180. * (rz)

    sx = tt.sin(rx)
    sy = tt.sin(ry)
    sz = tt.sin(rz)
    cx = tt.cos(rx)
    cy = tt.cos(ry)
    cz = tt.cos(rz)
    Rx = [[1,0,0],[0,cx,-sx],[0,sx,cx]]
    Ry = [[cy,0,sy],[0,1,0],[-sy,0,cy]]
    Rz = [[cz,-sz,0],[sz,cz,0],[0,0,1]]


    Rxt = tt.stacklists(Rx)
    Ryt = tt.stacklists(Ry)
    Rzt = tt.stacklists(Rz)
    full_rotation=tt.dot(Rzt,tt.dot(Ryt, Rxt))
    return full_rotation


def apply_trans():

    new_pos = (s/100.)*tt.dot(rot,pos)+(1./1000.)*t[:,np.newaxis]
    new_err = abs((s/100.)*tt.dot(rot,err))

def load_param_file(fName):

    if fName.endswith('.json'):
        with open(fName) as f:
            return json.load(f)

class Mirror(pm.Model):

    def __init__(self, definition_file, name='', model=None):
        self.definition = load_param_file(definition_file)
        try:
            if name == '':
                name = self.definition['default_name']
        except KeyError:
            pass
        super(Mirror, self).__init__(name, model)
        self.R = self.definition['geometry']['R']
        self.k = self.definition['geometry']['k']

def get_primary_model(pos, err, TMAP=None, model=None):

    if TMAP is None:
        TMAP = PrimaryTmap(1.,1.,1.,1.,1.,1.,0.,1.)
    print(TMAP)
    if model is None:
        model = pm.Model()
    with model:

        #Weak priors on translations and rotations. 

        #Translation variables:
        if TMAP['tx']:
            prim_tx = pm.Normal('tx', mu=0, sd=1000.)
        else:
            prim_tx = 0.
        if TMAP['ty']:
            prim_ty = pm.Normal('ty', mu=0, sd=1000.)
        else:
            prim_ty = 0.
        if TMAP['tz']:
            prim_tz = pm.Normal('tz', mu=0, sd=1000.)
        else:
            prim_tz = 0.
        #Rotation variables:
        if TMAP['rx']:
            prim_rx = pm.Uniform('rx', lower=-20., upper=20.)
        else:
            prim_rx = 0.
        if TMAP['ry']:
            prim_ry = pm.Uniform('ry', lower=-20., upper=20.)
        else:
            prim_ry = 0.
        if TMAP['rz']:
            prim_rz = pm.Uniform('rz', lower=-20., upper=20.)
        else:
            prim_rz=0.
        #Scale
        if TMAP['s']:
            prim_s = pm.Normal('s', mu=100, sd=10.)
        else:
            prim_s=100.
        #R - behaves a lot like scaling the mirror instead of scaling the dataset
        if TMAP['R']:
            prim_R = pm.Normal('R', mu=4400., sd=5)
        else:
            prim_R = 4400.

        prim_std = pm.Normal('std_intrinsic', mu=40., sd=30.)
        c = 1./R
        k = -1.
        target_thickness = .2
        prim_rot = theano_rot(rx,ry,rz)
        #roffset = theano.shared(np.array([0,2800.,0.]))
        prim_t = tt.stacklists([tx,ty,tz])
        prim_new_pos = (prim_s/100.)*tt.dot(prim_rot,pos)+(1./1000.)*prim_t[:,np.newaxis]
        prim_new_err = abs((s/100.)*tt.dot(rot,err))
        prim_rsq = new_pos[0]**2+new_pos[1]**2
        prim_sigmarsq = (new_err[0]**2*new_pos[0]+new_err[1]**2*new_pos[1])/rsq
        prim_con = (c*rsq)/(1.+tt.sqrt(1.-(1.+k)*c**2.*rsq))
        prim_a = tt.sqrt((R**2. - rsq*(k + 1.))/R**2.)
        prim_mrsq = (tt.sqrt(rsq)*(3.*R**2.*a*(a + 1.) + rsq*(k + 1.))/(R**3.*a*(a + 1.)**2.))**2
        prim_e_rescale = 1./(mrsq+1.)
        dist = (((new_pos[2]-con)*tt.sqrt(e_rescale)) - target_thickness)*1000.

        dist_error = 1000*tt.sqrt(new_err[2]**2*e_rescale + mrsq*e_rescale*sigmarsq)
        pm.Deterministic('std_measurement',tt.std(dist))
        test = pm.Normal('dist', mu=0, sd=tt.sqrt(dist_error**2+std**2), observed=dist)
        #test = pm.Normal('dist', mu=0, sd=50, observed=dist)

    return model

if __name__ == '__main__':

    print(mf.__file__)
    data = mf.dataset(from_file='20181208_primary_receiver.txt')
    print(data.ntarget)

    primary_points = data.subset_from_label('PRIMARY')
    primary = mf.dataset(points=primary_points)
    pos = primary.to_tensor()
    err = primary.to_err_tensor()
    TMAP = {'tx':True, 'ty':True, 'tz':True, 'rx':True, 'ry':True, 'rz':False, 's':False, 'R':True}
    test = Mirror(definition_file='POLARBEAR/SA_Primary.json')
    with pm.Model() as model:
        print(test)
    with test as model:
        r=22
    with test as model:
        test2 = test.test()
    with model:
        trace = pm.sample(2000, tune=1000, init = 'advi+adapt_diag', nuts_kwargs={'target_accept': .95, 'max_treedepth': 15})
    pm.traceplot(trace)
    plt.show()
    quit()
    model = get_primary_model(pos,err,TMAP=TMAP)
    with model:
        trace = pm.sample(2000, tune=1000, init = 'advi+adapt_diag', nuts_kwargs={'target_accept': .95, 'max_treedepth': 15})

    pm.save_trace(trace)
    print(trace)
    pm.traceplot(trace)
    #pm.traceplot(trace, varnames=['tx','ty', 'std'])
    plt.show()
