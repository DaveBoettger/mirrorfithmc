import numpy as np
import theano.tensor as tt
import theano
try:
    from mirrorfithmc.lib_transform import *
except:
    from lib_transform import *

def z_proj_dist_error(tds, R, k, translate_factor=1., target_thickness=0., return_error=True, full_errors=False, retspecial=False):
    '''Takes a point cloud DatasetTensor or DatasetArray object and mirror geometry paramters and returns the distances and errors
    from all the points in the cloud.
    '''
    if 'DatasetTensors' in type(tds).__name__:
        mlib = tt
    elif 'DatasetArrays' in type(tds).__name__:
        mlib = np

    if full_errors:
        Rsq = R**2
        rsq = tds.pos[0]**2+tds.pos[1]**2
        kp1 = (k+1)
        a = rsq*kp1
        b = mlib.sqrt(Rsq-a)
        
        conz = rsq / (R * (mlib.sqrt(1-a/Rsq)+1))
        S = b/mlib.sqrt(Rsq-k*rsq)
        dz0 = (tds.pos[2]-conz)
        dist = translate_factor*(S*dz0-target_thickness)

        if return_error:        
            absR = np.abs(R)
            rho = mlib.sqrt(rsq)
            gamma = (rho * Rsq) / (Rsq*(Rsq-rsq*(2*k+1))+rsq**2*k*kp1)
            slope = (rho*absR*(a+2*b*(b+absR))) / (R*b*(b+absR)**2)
            sigmarhosq= ((tds.pos[0]*tds.err[0])**2 + (tds.pos[1]*tds.err[1])**2) / (rsq)
            disterror = translate_factor*S*mlib.sqrt(tds.err[2]**2 + (gamma*dz0+slope)**2*sigmarhosq)

    else:
        Rsq = R**2
        rsq = tds.pos[0]**2+tds.pos[1]**2
        a = rsq*(k+1)
        c = Rsq-rsq*k

        conz = rsq / (R * (mlib.sqrt(1-a/Rsq)+1))
        dz0 = (tds.pos[2]-conz)
        S = mlib.sqrt(Rsq-a)/mlib.sqrt(c)
        dist = translate_factor*(S*dz0-target_thickness)

        if return_error:
            sigmarhosq= ((tds.pos[0]*tds.err[0])**2 + (tds.pos[1]*tds.err[1])**2) / (rsq)
            normzerosq = (rsq/c)
            disterror = translate_factor*mlib.sqrt(S**2*tds.err[2]**2+sigmarhosq*normzerosq)

    if return_error:
        if retspecial:
            return dist, disterror, sigmarhosq
        else :
            return dist, disterror
    else:
        return dist

def sim_conic_z_projection(k, R, XY, sigmaDelta=0., sigmaX=0., sigmaY=0., sigmaZ=0., target_thickness=.2):
    '''XY should be provided in local coordinates'''
    rsq = XY[0]**2+XY[1]**2
    rho = np.sqrt(rsq)
    Rsq = R**2
    kp1 = (k+1)
    
    a = rsq*(k+1)
    c = Rsq-rsq*k

    #conz gives our Z coordinates, so these will be points on the ellipse.
    conz = rsq / (R * (np.sqrt(1-a/Rsq)+1))

    #Now we need to find the normal vector at each point so we can generate the delta displacements
    #S is the z component of the normal
    z = np.sqrt(Rsq-a)/np.sqrt(c)
    normzero = np.sqrt(rsq/c)#This is the rho component of the normal vector
    #We need to convert the rho component into X and Y components.
    x = -XY[0]/rho * normzero
    y = -XY[1]/rho * normzero
    x[rho==0] = 0.
    y[rho==0] = 0.
    norm = np.array([x,y,z])
    conic = np.array([XY[0], XY[1], conz])

    #Simulate construction error by displacing along the normal
    delta = sigmaDelta*np.random.standard_normal(size=conic.shape[1])+target_thickness
    displaced_conic = conic+delta[np.newaxis]*norm

    sx = sigmaX * np.random.standard_normal(size=conic.shape[1])
    sy = sigmaY * np.random.standard_normal(size=conic.shape[1])
    sz = sigmaZ * np.random.standard_normal(size=conic.shape[1])

    displaced_conic[0] += sx
    displaced_conic[1] += sy
    displaced_conic[2] += sz

    return displaced_conic, delta, [sx,sy,sz]
