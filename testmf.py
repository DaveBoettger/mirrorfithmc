import pymc3 as pm
import numpy as np
import matplotlib.pyplot as plt
import theano.tensor as tt
import theano
import mf3.mirrorfit as mf
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

def theano_rot(rx,ry,rz):
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

def dataset_to_tensors(dataset):
    arr = dataset.toArray()
    errs = dataset.toErrArray()

    pos_tensor = tt._shared(arr.T)
    err_tensor = tt._shared(errs.T)
    return pos_tensor, err_tensor

def getRSqFun(r_off=0., angle_off=0.):
    '''Returns the radius to the origin of the symmetrized coordinate system given x,y in non-symmetrized coordinates'''
    def r(x,y):
        return (x-r_off*np.cos(angle_off))**2+(y-r_off*np.sin(angle_off))**2
    
    return r

def print_model_params(tx=0.,ty=0.,tz=0.,rx=0.,ry=0.,rz=0.,s=1.):
    
        R = 4400.

        c = 1./R
        k = -1.
        target_thickness = .2
        rot = theano_rot(rx,ry,rz)
        print('rot',rot.eval())
        roffset = theano.shared(np.array([0,2800.,0.]))
        t = tt.stacklists([tx,ty,tz])
        new_pos = s*tt.dot(rot,pos)+t[:,np.newaxis]
        rsq = new_pos[0]**2+new_pos[1]**2
        print('rsq',np.max(np.abs(rsq.eval())))
        con = (c*rsq)/(1.+tt.sqrt(1.-(1.+k)*c**2.*rsq))
        print('con',np.max(np.abs(con.eval())))
        #a = tt.sqrt((R**2. - rsq*(k + 1.))/R**2.)
        a=1.
        mr = tt.sqrt(rsq)*(2.*R**2.*a*(a + 1.) + rsq*(k + 1.))/(R**3.*a*(a + 1.)**2.)
        print('mr',np.max(np.abs(mr.eval())))
        scale = 1./(mr**2+1.)
        print('scale',np.max(np.abs(scale.eval())))
        dist = (((new_pos[2]-con)*tt.sqrt(scale)) - target_thickness)*1000.
        print('dist', np.max(np.abs(dist.eval())))
        #test = pm.Normal('dist', mu=0, sd=tt.sqrt(point_errors+std**2), observed=dist)

def get_model(pos, err):
    
    with pm.Model() as model:
        #Weak priors on translations and rotations. 
        tx = pm.Normal('tx', mu=0, sd=10)
        ty = pm.Normal('ty', mu=0, sd=10)
        tz = pm.Normal('tz', mu=0, sd=10)
        #R = pm.Normal('R', mu=4400., sd=50)
        R = 4400.
        std = pm.Normal('std_RV', mu=40, sd=30)
        rx = pm.Uniform('rx', lower=-np.pi/8., upper=np.pi/8.)
        ry = pm.Uniform('ry', lower=-np.pi/8., upper=np.pi/8.)
        #rz = pm.Uniform('rz', lower=0, upper=2*np.pi)
        #tx=0.
        #ty=0.
        #tz=0.
        #rx=0.
        #ry=0.
        rz=0.#rz is completely degenrate, so don't bother fitting this
        #s = pm.Normal('s', mu=1, sd=.1)
        s=1.

        #R = 4400.
        c = 1./R
        k = -1.
        target_thickness = .2
        rot = theano_rot(rx,ry,rz)
        roffset = theano.shared(np.array([0,2800.,0.]))
        t = tt.stacklists([tx,ty,tz])
        new_pos = s*tt.dot(rot,pos)+t[:,np.newaxis]
        new_err = abs(s*tt.dot(rot,err))
        rsq = new_pos[0]**2+new_pos[1]**2
        sigmarsq = (new_err[0]**2*new_pos[0]+new_err[1]**2*new_pos[1])/rsq
        con = (c*rsq)/(1.+tt.sqrt(1.-(1.+k)*c**2.*rsq))
        a = tt.sqrt((R**2. - rsq*(k + 1.))/R**2.)
        mrsq = (tt.sqrt(rsq)*(3.*R**2.*a*(a + 1.) + rsq*(k + 1.))/(R**3.*a*(a + 1.)**2.))**2
        scale = 1./(mrsq+1.)
        dist = (((new_pos[2]-con)*tt.sqrt(scale)) - target_thickness)*1000.

        dist_error = 1000*tt.sqrt(new_err[2]**2*scale + mrsq*scale*sigmarsq)
        pm.Deterministic('std',tt.std(dist))
        test = pm.Normal('dist', mu=0, sd=tt.sqrt(dist_error**2+std**2), observed=dist)
        #test = pm.Normal('dist', mu=0, sd=50, observed=dist)

    return model

if __name__ == '__main__':

    data=mf.dataset(from_file='../mirrorfit/POLARBEAR/pb_clouds/SA_NORTH_FIELD/20181208/aligned_cryostatrenamed.txt')  
    primary_points = [pp for pp in data if 'PRIMARY' in pp.label]
    primary = mf.dataset(points=primary_points)

    len_mds = len(primary)
    pos,err = dataset_to_tensors(primary)
    #for r in np.arange(-np.pi/8,np.pi/8.,.1):
    #    print_model_params(tx=10, ty=10, ry=r)
    #quit()
    model = get_model(pos,err)
    with model:
        trace = pm.sample(2000, tune=1000, init = 'advi+adapt_diag', nuts_kwargs={'target_accept': .95, 'max_treedepth': 15})

    pm.save_trace(trace)

    pm.traceplot(trace, varnames=['tx','ty', 'tz','rx', 'ry', 'std', 'std_RV'])
    #pm.traceplot(trace, varnames=['tx','ty', 'std'])
    plt.show()
