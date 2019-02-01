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

def theano_rot(rx,ry,rz, rescale=True):
    '''Return a theano tensor representing a rotation 
    matrix using specified rotation angles rx,ry, rz
    If rescale is True, treat the input angles as degrees.
    '''

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

class TheanoTransform():
    '''Define a transform object that can act on DatasetTenors and themselves via multiplication
    The operation returs either a new DatasetTensor or a new TheanoTransform object, depending
    on what is being operated on.
    Transforms can be defined using a trans dictionary with any or all  of the keys:
    ['tx', 'ty', 'tz', 'rx', 'ry', 'rz', 's'] for the translations, rotations, and scale.

    You can also pass tr, a vector of the three translations, R, a 3x3 rotation matrix, and s, the scale factor.
    Any of the final three override any setting in the dictionary.
    
    If no information is provided the identity transform is returned.
    '''
    def __init__(self, trans=None, tr=None, R=None, s=None, full_scale = 100., translate_factor=1000., rotation_in_degrees=True):

        self.full_scale = full_scale
        #translating by translate_factor units is equivalent to 1 data unit
        #typically this will be 1000 microns to 1 mm
        self.translate_factor = translate_factor 
        self.rotation_in_degrees = rotation_in_degrees
        #These reparameterizations help with the stability of the MCMC samples
        self._trans = trans
        self._tr = tr
        self._R = R
        self._s = s

        self._generate()

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
                except KeyError:
                    pass

        if self._R is None:
            self._R = theano_rot(rx=trans['rx'], ry=trans['ry'], rz=trans['rz'], rescale = self.rotation_in_degrees)

        if self._tr is None:
            self._tr = tt.stacklists([trans['tx'],trans['ty'],trans['tz']])

        if self._s is None:
            self._s = trans['s']

    def __mul__(self, other):
        '''Apply the transfrom'''
        
        if type(other) == DatasetTensors:
            pos = other.pos
            #In the function we treat error as a signed quantity to get the rotation correct,
            #but we set the actual .err value of the object we return to be
            #the absolute value of this since it represents the standard deviation of a Gaussian
            #There is probably a nicer algebraic way to do this (???) TODO
            err = other.serr
            new_pos = (self._s/self.full_scale)*tt.dot(self._R,pos)+(1./self.translate_factor)*self._tr[:,np.newaxis]
            new_err = ((self._s/self.full_scale)*tt.dot(self._R,err))

            #note Theano appropriately overloads the normal abs operator to work with their tensors
            return DatasetTensors(pos=new_pos, err=abs(new_err), serr=new_err)

        elif type(other) == TheanoTransform:
            #Apply to other transform and return a new transform equivalent to successivly applying the two transforms 
            new_tr = ((self._tr/self.translate_factor) + (self._s/self.full_scale)*tt.dot(self._R, (other._tr/other.translate_factor)))*self.translate_factor
            new_R = tt.dot(self._R, other._R)
            new_s = (self._s/self.full_scale * other._s/other.full_scale)*self.full_scale
            return TheanoTransform(tr=new_tr, R=new_R, s=new_s, full_scale=self.full_scale, translate_factor=self.translate_factor)

        else:
            raise NotImplementedError

    def __invert__(self): 
        '''Return the inverse of the translation (~t) such that t*~t is the identity'''
        new_R = pm.math.matrix_inverse(self._R)
        new_tr = -tt.dot(self._tr/self.translate_factor,new_R.T)/(self._s/self.full_scale)*self.translate_factor
        new_s = (1./(self._s/self.full_scale))*self.full_scale
        return TheanoTransform(tr=new_tr, R=new_R, s=new_s, full_scale=self.full_scale, translate_factor=self.translate_factor)