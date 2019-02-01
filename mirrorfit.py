import numpy as np
import theano.tensor as tt
from collections import namedtuple, OrderedDict
from lib_transform import *

class point(object):
    '''Represents a 3D point and associated gaussian error along with a label'''
    
    def __init__(self):
        self.pos = np.zeros(3)
        self.label = ""
        self.err = np.zeros(3)

    def read_ph_file_line(self,line):
        ll=line.split()
        # Check for unformatted data
        if len(ll) == 3:
            for i in range(3): self.pos[i] = float(ll[i])
        else:
            self.label = ll[0]
            for i in range(3): self.pos[i] = float(ll[i+1])
            if len(ll) > 4:
                for i in range(3): self.err[i] = np.abs(float(ll[i+4]))

    def __repr__(self):
        return 'POINT {0} - {1},{2}'.format(self.label,self.pos,self.err)
        
    def copy(self):
        p = point()
        p.pos = self.pos.copy()
        p.err = self.err.copy()
        p.label = self.label
        p.type = self.type
        return p

class dataset(OrderedDict):
    '''Represents a collection of point objects, usually representing all the points in a single measurement.
    
    This is designed for convenience, not speed. This should be used for manipulating the data
    prior to model creation.
    DatasetTensors should be used in loops or models.
    '''
    def __init__(self, points=None, from_file=None, markers=None):

        super().__init__()

        if markers is not None:
            self.markers = markers
        else:
            self.markers = []

        if points is not None:
            for point in points:
                self.add_point(point)
        if from_file is not None:
            self.read_data_file(from_file)

    def copy(self):
        c = dataset()
        for p in self.values(): c.add_point(p.copy())
        return c
        
    def add_point(self,point):
        self[point.label] = point

    def remove_point(self, point):
        del self[point.label]

    def read_data_file(self, filename):
        with open(filename) as f:
            for l in f:
                if l[0] == '*':
                    self.markers.append(l[1:].strip().upper())
                elif l[0]!="#":
                    p = point()
                    p.read_ph_file_line(l)
                    if p.label == "":
                        p.label = "TARGET_%d"%(self.ntarget+1)
                    self.add_point(p)

    def __getattr__(self, name):

        if name == 'markers':
            #ideally self.markers should always be defined as something, but if not
            #set it as an empty list and return it. This prevents infinite recursion
            newlist = []
            self.__setattr__('markers', newlist)
            return newlist

        elif name == 'labels':
            return [p.label for p in self.values()]
        elif name == 'pos':
            return [p.pos for p in self.values()]
        elif name == 'error' or name == 'err':
            return [p.err for p in self.values()]
        elif name.upper() in self.markers:
            return self.subset_from_marker(name.upper())
        else:
            raise AttributeError

    def __repr__(self):
        rep = '\n'.join([f'Dataset with {len(self)} points and {len(self.markers)} markers:',
            *[str(v) for v in self.values()]])
        if len(self.markers):
            rep = '\n'.join([rep, 'Recoginized markers:', *self.markers])
        return rep

    def to_array(self):
        '''Return the positions of points in the dataset as a numpy array'''
        return np.array(self.pos)

    def to_err_array(self):
        '''Return the errors of points in the dataset as a numpy array'''
        return np.array(self.error)

    def to_tensor(self):
        '''Return the positions of points in the dataset as a Theano tensor'''
        arr = self.to_array()
        return tt._shared(arr.T)

    def to_err_tensor(self):
        '''Return the positions of points in the dataset as a Theano tensor'''
        arr = self.to_err_array()
        return tt._shared(arr.T)

    def to_arrays(self):
        err = self.to_err_array()
        return DatasetArrays(pos=self.to_array(), err=err, serr=err)

    def to_tensors(self):
        err = self.to_err_tensor()
        return DatasetTensors(pos=self.to_tensor(), err=err, serr=err)

    def subset_from_labels(self, labels):
        '''Return a new dataset as a subset of self by exactly matching labels'''

        return dataset(points=[self[p] for p in labels if p in self])

    def subset_from_marker(self, marker):
        '''Return a new dataset as a subset of self by matching a search string IN the point labels'''

        return dataset(points=[self[p] for p in self if marker in p])

    def labels_in_common(self, other, marker=None):
        '''Return a list of labels with labels in common between this dataset and another,
        optionally using marker to further define the subset'''

        #can this be re-written with set notation over the dict_key object? 
        if marker is None:
            return [l for l in self if l in other and l]
        else:
            return [l for l in self if l in other and marker in l]

    def subsets_in_common(self, other, marker=None):
        '''Return a subset with labels in common between this dataset and another,
        optionally using marker to further define the subset
        Note that this has the added benefit of returning the two subsets in the same order,
        which can be useful when converting to arrays. 
        '''

        common_labels = self.labels_in_common(other, marker)
        return (self.subset_from_labels(common_labels), other.subset_from_labels(common_labels))

class AlignDatasetSimple(pm.Model):
    '''Find transform to apply to ds1 to match it to ds2
    Searches the datasets for common labels before aligning.
    '''
    def __init__(self, ds1, ds2, name='Alignment', fitmap=None, use_marker=None, model=None):
        super(AlignDatasetSimple, self).__init__(name, model)
        default_fitmap = {'tx':True, 'ty':True, 'tz':True, 'rx':True, 'ry':True, 'rz':True, 's':True}
        if fitmap is not None:
            default_fitmap.update(fitmap)
        self.fitmap = default_fitmap

        if use_marker is not None:
            ds1,ds2 = ds1.subsets_in_common(ds2, marker=use_marker)
        else:
            ds1,ds2 = ds1.subsets_in_common(ds2)

        self.ds1t=ds1.to_tensors()
        self.ds2t=ds2.to_tensors()

        print(f'fitmap is {self.fitmap}')
        #We need a better way of defining the priors here, needs thought TODO
        #Translation variables:
        tvals = {}
        if self.fitmap['tx']:
            tvals['tx'] = pm.Normal('tx', mu=0, sd=10000.)
        else:
            tvals['tx'] = 0.
        if self.fitmap['ty']:
            tvals['ty'] = pm.Normal('ty', mu=0, sd=10000.)
        else:
            tvals['ty'] = 0.
        if self.fitmap['tz']:
            tvals['tz'] = pm.Normal('tz', mu=0, sd=10000.)
        else:
            tvals['tz'] = 0.
        #Rotation variables:
        if self.fitmap['rx']:
            tvals['rx'] = pm.Normal('rx',mu=0, sd=200)
        else:
            tvals['rx'] = 0.
        if self.fitmap['ry']:
            tvals['ry'] = pm.Normal('ry',mu=0, sd=200)
        else:
            tvals['ry'] = 0.
        if self.fitmap['rz']:
            tvals['rz'] = pm.Normal('rz',mu=0, sd=200)
        else:
            tvals['rz'] = 0.
        #Scale
        if self.fitmap['s']:
            tvals['s'] = pm.Normal('s', mu=100, sd=10.)
        else:
            tvals['s'] = 100.
        #Compute the transform
        trans = TheanoTransform(trans=tvals)
        #apply the transform
        self.ds2tprime = trans*self.ds2t

        #Compute the distance between the two clouds and the error on that distance
        diff = pm.Deterministic('diff', self.ds1t.pos-self.ds2tprime.pos)
        variance = pm.Deterministic('variance', self.ds1t.err**2+self.ds2tprime.err**2)

        #do the alignment
        align = pm.Normal('align', mu=0, tau=1./variance, observed=diff)

    @classmethod
    def distances_from_trace(self, trace):
        varname = [v for v in trace.varnames if 'diff' in v][0]
        diff = trace.get_values(varname)
        return np.linalg.norm(diff, axis=1)

    @classmethod
    def distance_errors_from_trace(self, trace):
        varname = [v for v in trace.varnames if 'variance' in v][0]
        variance = trace.get_values(varname)
        #This metric does not vary over the trace so we cut out the trace dimesion
        return np.sqrt(np.linalg.norm(variance, ord=1, axis=1))[0,:]

    @classmethod
    def distances_and_errors_from_trace(self, trace):
        return (self.distances_from_trace(trace), self.distance_errors_from_trace(trace))
