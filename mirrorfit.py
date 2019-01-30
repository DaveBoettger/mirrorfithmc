import numpy as np
import theano.tensor as tt
from collections import namedtuple

DatasetArrays = namedtuple('DatasetArrays', ['pos', 'err'])

class point(object):
    def __init__(self):
        self.pos = np.zeros(3)
        self.label = ""
        self.type = ""
        self.err = np.zeros(3)

    def read_ph_file_line(self,line):
        ll=line.split()
        # Check for unformatted data
        if len(ll) == 3:
            self.type = "TARGET"
            for i in range(3): self.pos[i] = float(ll[i])
        else:
            self.label = ll[0]
            if "CODE" in self.label: self.type = "CODE"
            elif "TARGET" in self.label: self.type = "TARGET"
            elif "NUGGET" in self.label: self.type = "NUGGET"
            elif "CSB" in self.label: self.type = "CSB"
            else: self.type = "OTHER"
            for i in range(3): self.pos[i] = float(ll[i+1])
            if len(ll) > 4:
                for i in range(3): self.err[i] = float(ll[i+4])

    def __repr__(self):
        return 'POINT {0} - {1},{2}'.format(self.label,self.pos,self.err)
        
    def copy(self):
        p = point()
        p.pos = self.pos.copy()
        p.err = self.err.copy()
        p.label = self.label
        p.type = self.type
        return p

class dataset(dict):

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

    def getAttribute(self, att, sel = None):
        '''Legacy function to implement retrival of some attributes while also using a selection argument.
        Consider removing.
        '''

        if sel is None: sel = range(self.npoint)
        elif type(sel[0]) == bool or type(sel[0]) == np.bool_: sel = np.where(sel)[0]
        return np.array([self[p].__dict__[att] for p in sel])

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
        return DatasetArrays(pos=self.to_array(), err=self.to_err_array())

    def to_tensors(self):
        return DatasetArrays(pos=self.to_tensor(), err=self.to_err_tensor())

    def subset_from_labels(self, labels):
        '''Return a new dataset as a subset of self by exactly matching labels'''

        return dataset(points=[self[p] for p in self if p in labels])

    def subset_from_marker(self, marker):
        '''Return a new dataset as a subset of self by matching a search string IN the point labels'''

        return dataset(points=[self[p] for p in self if marker in p])

    def labels_in_common(self, other, marker=None):
        '''Return a list of labels with labels in common between this dataset and another,
        optionally using marker to further shrink the subset'''

        if marker is None:
            return [l for l in self if l in other and l]
        else:
            print(marker)
            return [l for l in self if l in other and marker in l]

    def subsets_in_common(self, other, marker=None):
        '''Return a subset with labels in common between this dataset and another,
        optionally using marker to further shrink the subset'''

        common_labels = self.labels_in_common(other, marker)
        return (self.subset_from_labels(common_labels), other.subset_from_labels(common_labels))
