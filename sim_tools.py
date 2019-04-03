import numpy as np

try:
    import mirrorfithmc.mirrorfithmc as mf
except:
    import mirrorfithmc as mf


def build_XY_circle(radius=1000., x0=0., y0=0., grid_count=10):
    
    xstep = ((radius-x0)-(-radius-x0)) / grid_count
    ystep = ((radius-y0)-(-radius-y0)) / grid_count
    x,y = np.mgrid[-radius-x0:radius-x0:xstep,-radius-y0:radius-y0:ystep]
    z = np.where(np.sqrt((x+x0)**2+(y+y0)**2)<radius,1,np.nan)
    x = x[z==1]
    y = y[z==1]
    return np.array([x.flatten(),y.flatten()])

def sim_ds_z_projection(k, R, XY, sigmaDelta=0., sigmaX=0., sigmaY=0., sigmaZ=0., target_thickness=.2, marker='SIMPOINT', name='SimulatedData'):
    conic, deltas, sigmas = mf.geometry.sim_conic_z_projection(k, R, XY, sigmaDelta, sigmaX, sigmaY, sigmaZ, target_thickness)
    points = []
    for i, (x,y,z, delta, sx, sy, sz) in enumerate(zip(conic[0,:], conic[1,:], conic[2,:], deltas, sigmas[0], sigmas[1], sigmas[2])):
        p = mf.point(pos=np.array([x,y,z]), err=np.array([sigmaX, sigmaY, sigmaZ]), label=f'{marker}{i+1}')
        p.metadata = f'delta={delta},sx={sx},sy={sy},sz={sz}'
        points.append(p)

    ds = mf.Dataset(points=points, name=name)
    return ds



