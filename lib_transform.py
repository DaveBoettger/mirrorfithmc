import numpy as np
import pymc3 as pm
import theano.tensor as tt
import theano
from collections import namedtuple
from theano.gof import MissingInputError

#DatasetArrays are named tuples of two np.ndarrays (the position and error tensors for a Dataset).
DatasetArrays = namedtuple('DatasetArrays', ['pos', 'err', 'serr'])
#DatasetTenors are named tuples of two Theano tensors (the position and error tensors for a Dataset).
DatasetTensors = namedtuple('DatasetTensors', ['pos', 'err', 'serr'])

def arrays_to_tensors(self, arrays):

    #err = self.to_err_tensor()
    #return DatasetTensors(pos=self.to_tensor(), err=err, serr=err)
    if type(arrays) == np.ndarray:
        return tt._shared(arrays.T)

    elif type(arrays) == DatasetArrays:
        return DatasetTensors(arrays_to_tensors(arrays.pos), arrays_to_tensors(arrays.err), arrays_to_tensors(serr))

def theano_rot(rx,ry,rz, rescale=180./np.pi):
    '''Return a theano tensor representing a rotation 
    matrix using specified rotation angles rx,ry, rz
    Rescale can be used to change the angular units to i.e. degrees.
    '''

    rx = (rx)/rescale
    ry = (ry)/rescale
    rz = (rz)/rescale

    sx = tt.sin(rx)
    sy = tt.sin(ry)
    sz = tt.sin(rz)
    cx = tt.cos(rx)
    cy = tt.cos(ry)
    cz = tt.cos(rz)
    Rx = [[1.,0.,0.],[0.,cx,-sx],[0.,sx,cx]]
    Ry = [[cy,0.,sy],[0.,1.,0.],[-sy,0.,cy]]
    Rz = [[cz,-sz,0.],[sz,cz,0.],[0.,0.,1.]]


    Rxt = tt.stacklists(Rx)
    Ryt = tt.stacklists(Ry)
    Rzt = tt.stacklists(Rz)
    full_rotation=tt.dot(Rzt,tt.dot(Ryt, Rxt))
    return full_rotation

def theano_matrix2euler(R, rescale=180./np.pi):
    '''Return equivalent rotations (in radians) around x, y, and z axes given a rotation matrix'''
    oz = tt.arctan2(R[1][0],R[0][0])
    oy = tt.arctan2(-R[2][0],tt.sqrt(R[2][1]**2 + R[2][2]**2))
    ox = tt.arctan2(R[2][1],R[2][2])
    ox = ox*rescale
    oy = oy*rescale
    oz = oz*rescale
    return (ox, oy, oz)

class TheanoTransform():
    '''Define a transform object that can act on DatasetTenors and themselves via multiplication
    The operation returns either a new DatasetTensor or a new TheanoTransform object, depending
    on what is being operated on.
    Transforms can be defined using a trans dictionary with any or all of the keys:
    ['tx', 'ty', 'tz', 'rx', 'ry', 'rz', 's'] for the translations, rotations, and scale.

    You can also pass tr, a vector of the three translations, R, a 3x3 rotation matrix, and s, the scale factor.
    Any of the final three override any corresponding setting in the dictionary.
    
    If no information is provided an identity transform is returned.

    Note apply_factors_to_trans=True indicates that the trans specification
    is in natural units and need to be rescaled to the working units.
    '''
    #translating by translate_factor units is equivalent to 1 data unit
    #typically this will be 1000 microns to 1 mm
    #These reparameterizations help with the stability of the MCMC samples
    translate_factor = 1000.
    #translate_factor = 100.
    full_scale = 100.
    rotation_scale = 180./np.pi

    def __init__(self, trans=None, tr=None, R=None, s=None, apply_factors_to_trans=False, rotation_center=None):
        self.reset_rotation_center(rotation_center)
        self._trans = trans
        self._tr = tr
        self._R = R
        self._s = s
        self._apply_factors_to_trans = apply_factors_to_trans
        self._generate()
        self.check_for_ident = False

    def reset_rotation_center(self, center=None):
        '''Set the origin of the rotation.
        If no center is provided, it is reset to the coordinate system origin.
        '''

        if center is None:
            self._rot_center = np.array([[0., 0., 0.]]).T 
        else:
            self._rot_center = np.array([center]).T

    def __repr__(self):
        '''Get a string representation of the tensor, trying to make it a useful one
        This is sort of assuming this will be used for debugging purposes only; this is pretty slow.
        '''
        try:
            try:
                tr=self._tr.eval()/self.translate_factor
            except MissingInputError:
                tr = theano.gof.op.get_test_value(self._tr)/self.translate_factor
                tr = f'{tr} (from test evaluation of analytic expression)'
        except:
            tr = self._tr/self.translate_factor

        try:
            try:
                R = self._R.eval()
            except MissingInputError:
                R = theano.gof.op.get_test_value(self._R)
                R = f'{R} (from test evaluation of analytic expression)'
        except:
            R = self._R

        try:
            try:
                s = self._s.eval()/self.full_scale
            except MissingInputError:
                s = theano.gof.op.get_test_value(self._s)/self.full_scale
                s = f'{s} (from test evaluation of analytic expression)'
        except:
            s = self._s/self.full_scale

        return f'effective translation: {tr}\nR: {R}\neffective scale factor: {s}'

    def _generate(self):
        '''Turn the specification of the transform into usable mathematical objects'''

        identity = {'tx':0., 'ty':0., 'tz':0., 'rx':0., 'ry':0., 'rz':0., 's':self.full_scale}
        trans = identity 
        if self._trans is not None:
            for k in identity.keys():
                try:
                    trans[k] = self._trans[k]
                    if self._apply_factors_to_trans:
                        if k in ['tx', 'ty', 'tz']:
                            trans[k]*=self.translate_factor
                        elif k in ['rx', 'ry', 'rz']:
                            trans[k]*=self.rotation_scale
                        elif k == 's':
                            trans[k]*=self.full_scale
                except KeyError:
                    pass

        if self._R is None:
            self._R = theano_rot(rx=trans['rx'], ry=trans['ry'], rz=trans['rz'], rescale = self.rotation_scale)

        if self._tr is None:
            self._tr = tt.stacklists([trans['tx'],trans['ty'],trans['tz']])

        if self._s is None:
            self._s = trans['s']

    def __mul__(self, other):
        '''Apply the transfrom'''
        
        if self.check_for_ident:
            if self.is_identity():
                return other
            elif type(other) == TheanoTransform and other.is_identity():
                return self

        if type(other) == DatasetTensors:
            pos = other.pos
            #In this function we treat error as a signed quantity to get the rotation correct,
            #but we set the actual .err value of the object we return to be
            #the absolute value of this since it represents the standard deviation of a Gaussian
            #There is probably a nicer algebraic way to do this (???) TODO
            err = other.serr
            new_pos = (self._s/self.full_scale)*tt.dot(self._R,(pos-self._rot_center))+self._rot_center+(1./self.translate_factor)*self._tr[:,np.newaxis]
            new_err = ((self._s/self.full_scale)*tt.dot(self._R,err))

            #note Theano appropriately overloads the normal abs operator to work with their tensors
            return DatasetTensors(pos=new_pos, err=abs(new_err), serr=new_err)

        elif type(other) == TheanoTransform:


            if np.sum(self._rot_center==0.)!=3 or np.sum(other._rot_center==0.)!=3:
                raise NotImplementedError('Transform composition is not supported when the rotation center is not the coordinate system origin.')
            #Apply to other transform and return a new transform equivalent to successivly applying the two transforms 
            new_tr = ((self._tr/self.translate_factor) + (self._s/self.full_scale)*tt.dot(self._R, (other._tr/other.translate_factor)))*self.translate_factor
            new_R = tt.dot(self._R, other._R)
            new_s = (self._s/self.full_scale * other._s/other.full_scale)*self.full_scale
            return TheanoTransform(tr=new_tr, R=new_R, s=new_s)

        elif type(other) == float or type(other) == int:
            if other == 1.:
                return self
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError

    def __invert__(self): 
        '''Return the inverse of the translation (~t) such that t*~t is the identity'''
        if np.sum(self._rot_center==0.)!=3:
            raise NotImplementedError('Transform inversion is not supported when the transform center is not the origin.')
        new_R = pm.math.matrix_inverse(self._R)
        new_tr = -tt.dot(self._tr/self.translate_factor,new_R.T)/(self._s/self.full_scale)*self.translate_factor
        new_s = (1./(self._s/self.full_scale))*self.full_scale
        return TheanoTransform(tr=new_tr, R=new_R, s=new_s)

    def get_final_trans(self):
        '''Return equivalent of a trans dictionary that represents the behavior of the object.
        The point of this function is that it provides a method of extracting the 
        transform parameters in a way that is independent of the parameters (trans, tr, or R) used 
        to originally define the transform.
        '''
        if np.sum(self._rot_center==0.)!=3:
            raise NotImplementedError('Transform parameter recovery is not supported when the transform center is not the origin.')
        tx = self._tr[0]
        ty = self._tr[1]
        tz = self._tr[2]
        rx,ry,rz = theano_matrix2euler(self._R)
        s = self._s
        return {'tx':tx, 'ty':ty, 'tz':tz, 'rx':rx, 'ry':ry, 'rz':rz, 's':s}

    def is_identity(self):
        '''Returns True if this transform is equal to the identity. This is slow and should only be used for setup.'''
        ident_final = TheanoTransform().get_final_trans()
        this_final=self.get_final_trans()
        for k in this_final:
            if type(this_final[k]) == tt.TensorVariable:
                try:
                    if this_final[k].eval() == ident_final[k].eval():
                        continue
                    else:
                        return False
                except MissingInputError:
                    return False
            else:
                if this_final[k] == ident_final[k]:
                    continue
                else:
                    return False
        return True
            
    def eval_on_dataset(self, ds, name_modifier=''):
        '''Convenience function to apply a transform to a dataset.

        For this to work it must be possible to call .eval() on the transformed coordinates.
        This means that the transform cannot be composed of variables, which is why this is not implemented in __mul__.
        '''
        dst = ds.to_tensors()
        dstp = self*dst
        dstafinal = DatasetArrays(pos=dstp.pos.eval(), err=dstp.err.eval(), serr=dstp.serr.eval())
        return ds.remake_from_arrays(dstafinal, name_modifier=name_modifier)
        
