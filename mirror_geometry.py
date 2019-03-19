import numpy as np
import theano.tensor as tt
import theano

def zProjDistError(tds, R, k, translate_factor=1., target_thickness=0., full_errors=False):

    if full_errors:
        absR = abs(R)
        Rsq = R**2
        rsq = tds.pos[0]**2+tds.pos[1]**2
        rho = tt.sqrt(rsq)
        kp1 = (k+1)
        a = rsq*kp1
        b = tt.sqrt(Rsq-a)
        conz = rsq / (R * (tt.sqrt(1-a/Rsq)+1))
        S = b/tt.sqrt(Rsq-k*rsq)
        gamma = (rho * Rsq) / (Rsq*(Rsq-rsq*(2*k+1))+rsq**2*k*kp1)
        slope = (rho*absR*(a+2*b*(b+absR))) / (R*b*(b+absR)**2)
        dz0 = (tds.pos[2]-conz)
        dist = translate_factor*(S*dz0-target_thickness)
        sigmarhosq= (tds.pos[0]*tds.err[0]+ tds.pos[1]*tds.err[1]) / (rsq)
        disterror = translate_factor*S*tt.sqrt(tds.err[2]**2 + (gamma*dz0+slope)**2*sigmarhosq)
    else:
        Rsq = R**2
        rsq = tds.pos[0]**2+tds.pos[1]**2
        a = rsq*(k+1)
        c = Rsq-rsq*k
        conz = rsq / (R * (tt.sqrt(1-a/Rsq)+1))
        dz0 = (tds.pos[2]-conz)
        S = tt.sqrt(Rsq-a)/tt.sqrt(c)
        sigmarhosq= (tds.pos[0]*tds.err[0]+ tds.pos[1]*tds.err[1]) / (rsq)
        normzerosq = (rsq/c)**2
        dist = translate_factor*(S*dz0-target_thickness)
        disterror = translate_factor*tt.sqrt(S**2*tds.err[2]**2+sigmarhosq*normzerosq)

    return dist, disterror
