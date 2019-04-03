import numpy as np
import theano.tensor as tt
import theano
from collections import namedtuple, OrderedDict
try:
    from mirrorfithmc.lib_transform import *
    import mirrorfithmc.util as util
    import mirrorfithmc.mirror_geometry as geometry
    import mirrorfithmc.plotting as plotting
except:
    from lib_transform import *
    import mirror_geometry as geometry
    import util as util
    import plotting as plotting

class point(object):
    '''Represents a 3D point and associated gaussian error along with a label'''
    
    def __init__(self, pos=None, err=None, label=None):
        if pos is None:
            self.pos = np.zeros(3)
        else:
            self.pos = np.array(pos)

        if label is None:
            self.label = 'UnnamedPoint'
        else:
            self.label = label

        if err is None:
            self.err = np.zeros(3)
        else:
            self.err = err

        self.metadata=None

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
        return p

class Dataset(OrderedDict):
    '''Represents a collection of point objects, usually representing all the points in a single measurement.
    
    This is designed for convenience, not speed. This should be used for manipulating the data
    prior to model creation.
    DatasetTensors should be used in loops or models.
    '''
    def __init__(self, points=None, from_file=None, markers=None, name=None, metadata=''):

        super().__init__()


        self.metadata = metadata

        if markers is not None:
            self.markers = markers
        else:
            self.markers = []

        if points is not None:
            for point in points:
                self.add_point(point)

        if from_file is not None:
            self.read_data_file(from_file)

        if name is None:
            try:
                self.name #may have been named in file
            except:
                self.name = 'Unnamed'
        else:
            self.name = name
        
    def copy(self, name_modifier=''):
        c = Dataset(name=f'{self.name}{name_modifier}')
        for p in self.values(): c.add_point(p.copy())
        return c
        
    def add_point(self,point):
        try:
            self[point.label]
        except KeyError:
            self[point.label] = point
        else:
            raise ValueError(f'Label "{point.label}" already exists!')

    def remove_point(self, point):
        del self[point.label]

    def read_data_file(self, filename):
        with open(filename) as f:
            for l in f:
                if l[0] == '#':
                    if l[1] == '*':
                        self.markers.append(l[2:].strip().upper())
                    if l[1:].startswith('@NAME='):
                        self.name = l.split('=')[1].strip()
                    if l[1:].startswith('@DSMETA:'):
                        lsplit = ':'.join(l.split(':')[1:])
                        self.metadata = lsplit
                    if l[1:].startswith('@POINTMETA:'):
                        lsplit = ':'.join(l.split(':')[1:]).split('=')
                        pname = lsplit[0]
                        pdata = '='.join(lsplit[1:])
                        try:
                            self[pname].metadata = pdata.strip()
                        except KeyError:
                            pass
                else:
                    p = point()
                    p.read_ph_file_line(l)
                    if p.label == "":
                        p.label = "TARGET_%d"%(self.ntarget+1)
                    self.add_point(p)

    def write_data_file(self, filename):
        with open(filename, 'w') as f:
            f.write(f'#@NAME={self.name}\n')
            if self.metadata != '':
                f.write(f'#@DSMETA:{self.metadata}\n')
            for p in self.values():
                f.write(f'{p.label}\t{p.pos[0]} {p.pos[1]} {p.pos[2]} {p.err[0]} {p.err[1]} {p.err[2]}\n')
                if p.metadata is not None:
                    f.write(f'#@POINTMETA:{p.label}={p.metadata}\n')

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
        return np.array(self.pos).T

    def to_err_array(self):
        '''Return the errors of points in the dataset as a numpy array'''
        return np.array(self.error).T

    def to_tensor(self):
        '''Return the positions of points in the dataset as a Theano tensor'''
        arr = self.to_array()
        return tt._shared(arr)

    def to_err_tensor(self):
        '''Return the positions of points in the dataset as a Theano tensor'''
        arr = self.to_err_array()
        return tt._shared(arr)

    def to_arrays(self):
        err = self.to_err_array()
        return DatasetArrays(pos=self.to_array(), err=err, serr=err)

    def to_tensors(self):
        err = self.to_err_tensor()
        return DatasetTensors(pos=self.to_tensor(), err=err, serr=err)

    def remake_from_tensors(self):
        pass

    def remake_from_arrays(self, arrays, name_modifier=''):

        pos = arrays.pos.T
        serr = arrays.serr.T
        labels = self.keys()
        new_ds = Dataset(name=f'{self.name}{name_modifier}')
        for p, e, l in zip(pos, serr, labels):
            new_point = point(pos=p, err=e, label=l)
            new_ds.add_point(new_point)
        return new_ds

    def subset_from_labels(self, labels, name_modifier=None):
        '''Return a new dataset as a subset of self by exactly matching labels
        Points are returned in the order labels are provided
        '''

        if name_modifier is None:
            name_modifier = ''
        return Dataset(points=[self[p] for p in labels if p in self], name=f'{self.name}{name_modifier}', metadata=self.metadata)

    def subset_from_marker(self, marker, name_modifier=None):
        '''Return a new dataset as a subset of self by matching a search string IN the point labels'''
        if name_modifier is None:
            #name_modifier = f'[{marker}]'
            name_modifier = '' 
        return Dataset(points=[self[p] for p in self if marker in p], name=f'{self.name}{name_modifier}', metadata=self.metadata)

    def labels_in_common(self, others, marker=None):
        '''Return a list of labels with labels in common between this dataset and others,
        optionally using marker to further refine the subset'''

        if type(others) == Dataset:
            others = [others]
    
        #can this be re-written with set notation over the dict_key object? 
        common_labels = set(self).intersection(*others)
        if marker is None:
            return common_labels
        else:
            return [l for l in common_labels if marker in l]

    def subsets_in_common(self, others, marker=None):
        '''Return a subset with labels in common between this dataset and another,
        optionally using marker to further define the subset
        Note that this has the added benefit of returning the two subsets in the same order,
        which can be useful when converting to arrays. 
        '''

        if type(others) == Dataset:
            others = [others]
        common_labels = self.labels_in_common(others, marker)
        if marker is not None:
            #name_modifier = f'[{marker}]'
            name_modifier = ''
        else:
            name_modifier = None
        datasets = [self]
        datasets.extend(list(others))

        return tuple([ds.subset_from_labels(common_labels, name_modifier=name_modifier) for ds in datasets])

    def get_center(self):
        '''Find the mean of all the points.

        This should eventually have the option of weighting by error.
        '''
        ar = self.to_array().T
        return np.array([np.mean(ar[0,:]), np.mean(ar[1,:]),np.mean(ar[2,:])])

    def get_spread(self):
        ar = self.to_array().T
        return np.array([np.max(ar[0,:])-np.min(ar[0,:]),np.max(ar[1,:])-np.min(ar[1,:]),np.max(ar[2,:])-np.min(ar[2,:])])

class AlignDatasets(pm.Model):
    '''Find transform to apply to ds1 to match it to ds2.

    fitmap can be used to specifiy which parts of the transform are used in the alignment.
    select_subsets=True means the datasets will be searched for common labels before aligning.
    use_marker='MARKER' will select only points with the indicated marker. This only works if select_subsets is True.

    error_scale1 is used to scale the errors in dataset 1 if error rescaling is turned on in the fitmap. A default will be generated if not provided.
    error_scale2 is used to scale the errors in dataset 2 if error rescaling is turned on in the fitmap. A default will be generated if not provided.
    '''
    def __init__(self, ds1, ds2, name=None, fitmap=None, select_subsets=True, use_marker=None, rotate_from_center = True, error_scale1=None, error_scale2=None, model=None):
        super(AlignDatasets, self).__init__(name, model)
        default_fitmap = {'tx':True, 'ty':True, 'tz':True, 'rx':True, 'ry':True, 'rz':True, 's':True, 'rescale_errors':False}
        if fitmap is not None:
            default_fitmap.update(fitmap)
        self.fitmap = default_fitmap

        if select_subsets:
            if use_marker is not None:
                ds1,ds2 = ds1.subsets_in_common(ds2, marker=use_marker)
            else:
                ds1,ds2 = ds1.subsets_in_common(ds2)
        self.ds1=ds1
        self.ds2=ds2
        self.ds1t=ds1.to_tensors()
        self.ds2t=ds2.to_tensors()
        if name is None:
            self.name = f'Align_{ds2.name}_to_{ds1.name}'

        #build up some simple priors
        center_diff = ds1.get_center()-ds2.get_center()
        max_spread = ds1.get_spread()+ds2.get_spread()
        if not rotate_from_center:
            max_spread += np.sum(np.abs(ds1.get_center()))
        mus = {'tx': center_diff[0], 'ty': center_diff[1], 'tz': center_diff[2]}
        sds = {'tx': max_spread[0], 'ty': max_spread[1], 'tz': max_spread[2]}
        print(f'{self.name} fitmap is {self.fitmap}')
        self.tvals = util.generate_standard_transform_variables(self.fitmap, mus=mus, sds=sds)
        print(f'TVALS are: {self.tvals}')
        #Error scale
        if self.fitmap['rescale_errors']:
            if error_scale1 is None:
                self.error_scale1 = util.generate_error_scale_distribution(name=f'error_scale_{ds1.name}')
            else:
                self.error_scale1 = error_scale1
            if error_scale2 is None:
                self.error_scale2 = util.generate_error_scale_distribution(name=f'error_scale_{ds2.name}')
            else:
                self.error_scale2 = error_scale2
        else:
            '''error_scale = 1 is equivalent to turning error scaling off.'''
            self.error_scale1 = 1.
            self.error_scale2 = 1.
        #Compute the transform
        self.trans = TheanoTransform(trans=self.tvals)

        if rotate_from_center:
            #find center of ds2
            ds2center = self.ds2.get_center()
            #rotate from center
            self.trans.reset_rotation_center(ds2center)
        #apply the transform
        self.ds2tprime = self.trans*self.ds2t

        #Compute the distance between the two clouds and the error on that distance
        self.diff =  self.ds1t.pos-self.ds2tprime.pos
        self.sd = tt.sqrt(((self.error_scale1*self.ds1t.err)**2+(self.error_scale2*self.ds2tprime.err)**2))

        #specify the alignment
        align = util.generate_alignment_distribution('align', sd=self.sd, observed=self.diff)

    def calc_diff(self, trace):
        '''Calculate the vector differences of the points and their standard deviation estimated from 
        the errors in the datasets'''
        sds = []
        diffs = []

        diff = theano.function(inputs=list(self.vars), outputs=self.diff, on_unused_input='ignore')
        sd = theano.function(inputs=list(self.vars), outputs=self.sd, on_unused_input='ignore')
        for p in trace.points():
            diffs.append(diff(**p))
            sds.append(sd(**p))
        return np.array(diffs).T, np.sqrt(np.array(sds)).T

    def use_transform_trace(self, other, trace):
        '''Builds functions that apply the transform from this fit to a DatasetTensors object.
        Return signed errors.
        '''

        fvals = self.trans*other

        posval = theano.function(inputs=list(self.vars), outputs=fvals.pos, on_unused_input='ignore')
        errval = theano.function(inputs=list(self.vars), outputs=fvals.serr, on_unused_input='ignore')

        poses = []
        errors = []
        for p in trace.points():
            poses.append(posval(**p).T)
            errors.append(errval(**p).T)

        return np.array(poses), np.array(errors)

    def mean_transform(self, trace):
        '''
        Return the expectation value of the transform parameters along with the standard deviations of the parameters
        This provides no information about correlations between the parameters
        '''

        fixed_tvals = {}
        uncertainties = {}
        for key in self.tvals:
            if type(self.tvals[key]) == pm.model.FreeRV:
                fixed_tvals[key] = np.mean(trace.get_values(self.tvals[key].name))
                uncertainties[key] = np.std(trace.get_values(self.tvals[key].name))
            else:
                fixed_tvals[key] = self.tvals[key]
        return TheanoTransform(fixed_tvals), uncertainties

class AlignManyDatasets(pm.Model):
    '''Align more than one dataset at a time.

    reference is a single dataset to which does not move. All others will be aligned to this.
    datasets is a list of datasets, all of which will be aligned to reference and to each other at the reference location.
    '''
    def __init__(self, reference, datasets, name=None, fitmaps=None, select_subsets=True, use_marker=None, scale_ref_error=False, model=None):
        super(AlignManyDatasets, self).__init__(name, model)

        if name is None:
            self.name='MultiAlign'
        else:
            self.name=name

        if select_subsets:
            #Find common points between all three sets and further shrink down using marker if provided
            if use_marker is not None:
                reference, *datasets = reference.subsets_in_common(datasets, marker=use_marker)
            else:
                reference, *datasets = reference.subsets_in_common(datasets)

        self.reference = reference
        self.datasets = datasets

        #Build as many fitmaps as we have datasets
        default_fitmap = {'tx':True, 'ty':True, 'tz':True, 'rx':True, 'ry':True, 'rz':True, 's':True, 'rescale_errors':False}
        if fitmaps is not None:
            if type(fitmaps) == 'dict':
                self.fitmaps = [default_fitmap.copy().update(fitmaps)]*len(datasets)
            else:
                self.fitmaps = [default_fitmap.copy().update(fitmap) for fitmap in fitmaps]
        else:
            self.fitmaps = [default_fitmap]*len(datasets)
        
        self.error_scales = []
        if scale_ref_error:
            self.error_scales.append(util.generate_error_scale_distribution(name=f'error_scale_{reference.name}'))
        else:
            self.error_scales.append(1.)
        
        for ds, fm in zip(self.datasets, self.fitmaps):
            if fm['rescale_errors']:
                self.error_scales.append(util.generate_error_scale_distribution(name=f'error_scale_{ds.name}'))
            else:
                self.error_scales.append(1.)

        self.alignments = []
        temp_alignments = []
        #Align everything to the reference set
        for ds, fm, es in zip(self.datasets, self.fitmaps, self.error_scales[1:]):
            temp_alignments.append(AlignDatasets(ds1=reference, ds2=ds, use_marker=use_marker, fitmap=fm, select_subsets=False, error_scale1 = self.error_scales[0], error_scale2 = es))

        self.cross_diffs = {}
        self.cross_sds = {}
        #relate datasets to each other, insisting that since all the datasets are aligned to the reference,
        #they must also now all be aligned to each other.
        while len(temp_alignments)>1:
            temp1 = temp_alignments.pop(0) #pop out alignment object (we pop(0) to avoid reversing the order of the list)
            self.alignments.append(temp1) #move this into a final list of alignments  
            for temp2 in temp_alignments:
                key_name = f'{temp1.ds2.name}-{temp2.ds2.name}'
                #Compute the distance between the two clouds and the error on that distance
                #Both are datasets that have already had a transformation applied to align them to the reference set
                diff = temp1.ds2tprime.pos-temp2.ds2tprime.pos
                sd = tt.sqrt(((temp1.error_scale2*temp1.ds2tprime.err)**2+(temp2.error_scale2*temp2.ds2tprime.err)**2))
                #store the symbolic representation of these variables so they can be reconstructed from a trace
                self.cross_diffs[key_name] = diff 
                self.cross_sds[key_name] = sd 
                #specify the alignment
                align = util.generate_alignment_distribution(f'align_{key_name}', sd=sd, observed=diff)
        #We now have one alignment object left in our temporary list, which we can move into our stored list
        self.alignments.append(temp_alignments.pop())

    def calc_diff(self, trace):
        '''Calculate the vector differences of the points and their standard deviation estimated from 
        the errors in the datasets
        This is computed between all combinations of datasets.
        '''
        sds = {}
        diffs = {} 

        testpoint = self.test_point
        #reference alignments:
        for align in self.alignments:
            keyname = f'{align.ds1.name}-{align.ds2.name}'
            sds[keyname] = []
            diffs[keyname] = []

            #diff = util.make_theano_function(align.diff)
            diff = theano.function(inputs=self.vars, outputs=align.diff, on_unused_input='ignore')
            sd = theano.function(inputs=self.vars, outputs=align.sd, on_unused_input='ignore')
            for p in trace.points():
                p2 = dict([(k,v) for k,v in p.items() if k in testpoint.keys()])
                diffs[keyname].append(diff(**p2))
                sds[keyname].append(sd(**p2))
            diffs[keyname] = np.array(diffs[keyname])
            sds[keyname] = np.array(sds[keyname])
        #cross alignments:
        for ckey in self.cross_diffs:
            sds[ckey] = []
            diffs[ckey] = []
            diff = theano.function(inputs=list(self.vars), outputs=self.cross_diffs[ckey], on_unused_input='ignore')
            sd = theano.function(inputs=list(self.vars), outputs=self.cross_sds[ckey], on_unused_input='ignore')
            for p in trace.points():
                p2 = dict([(k,v) for k,v in p.items() if k in testpoint.keys()])
                diffs[ckey].append(diff(**p2))
                sds[ckey].append(sd(**p2))

            diffs[ckey] = np.array(diffs[ckey])
            sds[ckey] = np.array(sds[ckey])

        return (diffs, sds)

class AlignMirror(pm.Model):
    '''Find transform to apply to ds1 to match it to the mirror given py mirror_definition

    fitmap can be used to specifiy which parts of the transform are used in the alignment.
    use_marker='MARKER' will select only points with the indicated marker.
    '''

    def __init__(self, ds, mirror_definition, name=None, fitmap=None, use_marker=None, target_thickness = .2, model=None):
        super(AlignMirror, self).__init__(name, model)

        self.definition = util.load_param_file(mirror_definition)
        self.mirror_name = self.definition["default_name"]
        self.target_thickness = target_thickness

        default_fitmap = {'tx':True, 'ty':True, 'tz':True, 'rx':True, 'ry':True, 'rz':True, 's':False, 'R':True, 'mirror_std':True}

        for k in fitmap:
            if k not in default_fitmap:
                print(f'Warning: item {k} appears in fitmap but is not used for this alignment.')
        if fitmap is not None:
            default_fitmap.update(fitmap)
        self.fitmap = default_fitmap

        if use_marker is not None:
            ds = ds.subset_from_marker(use_marker)

        self.ds=ds
        self.dst=ds.to_tensors()

        if name is None:
            self.name = f'Align_{ds.name}_to_{self.mirror_name}'

        print(f'{self.name} fitmap is {self.fitmap}')
        self.tvals = util.generate_standard_transform_variables(fitmap)

        self.trans = TheanoTransform(trans=self.tvals)

        #apply the transform
        self.dstprime = self.trans*self.dst

        #Determine how we represent intrisic error on mirror (ie not measurement error but construction error)
        #(This is the prior on the intrinsic error)
        intrinsic_error = self.definition['errors']['surface_std']
        if fitmap['mirror_std']:

            std_bound = pm.Bound(pm.Normal,lower=0.0)
            spread_on_error = self.definition['errors']['surface_std_error'] #this parameter defines uncertainty on the intrinsic standard deviation
            self.std_intrinsic = std_bound(f'std_{self.mirror_name}_intrinsic', mu=intrinsic_error, sd=spread_on_error)
        else:
            self.std_intrinsic = intrinsic_error

        self.R = self.definition['geometry']['R']
        if fitmap['R']:
            self.R = pm.Normal('R', mu=self.R, sd=self.definition['errors']['deltaR'])

        c = 1./self.R
        k = self.definition['geometry']['k']

        rsq = self.dstprime.pos[0]**2+self.dstprime.pos[1]**2
        sigmarsq = ((self.dstprime.err[0]*self.dstprime.pos[0])**2+(self.dstprime.err[1]*self.dstprime.pos[1])**2)/rsq
        con = (c*rsq)/(1.+tt.sqrt(1.-(1.+k)*c**2*rsq))
        a = tt.sqrt((self.R**2. - rsq*(k + 1.))/self.R**2)
        mrsq = (tt.sqrt(rsq)*(2.*self.R**2.*a*(a + 1.) + rsq*(k + 1.))/(self.R**3.*a*(a + 1.)**2.))**2
        e_rescale = 1./(mrsq+1.)
        #The distance and the error on the distance are converted to microns (or whatever scale is used in the transform).
        self.dist = (((self.dstprime.pos[2]-con)*tt.sqrt(e_rescale)) - target_thickness)*self.trans.translate_factor
        self.dist_error = tt.sqrt(self.dstprime.err[2]**2*e_rescale + mrsq*e_rescale*sigmarsq)*self.trans.translate_factor
        pm.Deterministic('mirror_dist', self.dist)
        pm.Deterministic('mirror_dist_error', self.dist_error)
        #Specify the alignment
        align = util.generate_alignment_distribution(name='alignment', nu=5, sd=tt.sqrt(self.dist_error**2+self.std_intrinsic**2), observed=self.dist)

class AlignMirror2(pm.Model):
    '''Find transform to apply to ds1 to match it to the mirror given py mirror_definition

    fitmap can be used to specifiy which parts of the transform are used in the alignment.
    use_marker='MARKER' will select only points with the indicated marker.
    '''

    def __init__(self, ds, mirror_definition, name=None, fitmap=None, use_marker=None, use_errors=False, target_thickness = .2, model=None):
        super(AlignMirror2, self).__init__(name, model)

        self.definition = util.load_param_file(mirror_definition)
        self.mirror_name = self.definition["default_name"]
        self.target_thickness = target_thickness

        default_fitmap = {'tx':True, 'ty':True, 'tz':True, 'rx':True, 'ry':True, 'rz':True, 's':False, 'R':True, 'mirror_std':True}

        self.fitmap = util.update_fitmap(fitmap, default_fitmap)

        if use_marker is not None:
            ds = ds.subset_from_marker(use_marker)
        self.ds=ds
        self.dst=ds.to_tensors()

        if name is None:
            self.name = f'Align_{ds.name}_to_{self.mirror_name}'

        print(f'{self.name} fitmap is {self.fitmap}')
        self.tvals = util.generate_standard_transform_variables(fitmap)
        self.trans = TheanoTransform(trans=self.tvals)

        #apply the transform
        self.dstprime = self.trans*self.dst

        #Determine how we represent intrisic error on mirror (ie not measurement error but construction error)
        #(This is the prior on the intrinsic error)
        intrinsic_error = self.definition['errors']['surface_std']
        if fitmap['mirror_std']:
            std_bound = pm.Bound(pm.Normal,lower=0.0)
            spread_on_error = self.definition['errors']['surface_std_error'] #this parameter defines uncertainty on the intrinsic standard deviation
            self.std_intrinsic = std_bound(f'std_{self.mirror_name}_intrinsic', mu=intrinsic_error, sd=spread_on_error)
        else:
            self.std_intrinsic = intrinsic_error

        self.R = self.definition['geometry']['R']
        if fitmap['R']:
            self.R = pm.Normal('R', mu=self.R, sd=self.definition['errors']['deltaR'])

        self.k = self.definition['geometry']['k']

        if use_errors:
            self.dist, self.disterrors = geometry.z_proj_dist_error(self.dstprime, self.R, self.k, translate_factor=self.trans.translate_factor, target_thickness=target_thickness) 
            pm.Deterministic('mirror_dist_error', self.dist_error)
        else:
            self.dist = geometry.z_proj_dist_error(self.dstprime, self.R, self.k, return_error=False, translate_factor=self.trans.translate_factor, target_thickness=target_thickness) 
        
        pm.Deterministic('mirror_dist', self.dist)

        #Specify the alignment
        use_nu=np.inf
        if use_errors:
            self.align = util.generate_alignment_distribution(name='alignment', nu=use_nu, sd=tt.sqrt(self.dist_error**2+self.std_intrinsic**2), observed=self.dist)
        else:
            self.align = util.generate_alignment_distribution(name='alignment', nu=use_nu, sd=self.std_intrinsic, observed=self.dist)

