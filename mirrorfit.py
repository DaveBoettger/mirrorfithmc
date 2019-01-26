import numpy as np

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

class dataset(list):

    def __init__(self, points=None, from_file=None):
        super().__init__()
        #self.trnsf = transform_history()
        if points is not None:
            if isinstance(points[0],point):
                self.extend(points)
            else:
                for x in points:
                    p = point()
                    p.pos=x
                    p.type = "OTHER"
                    self.append(p)
        if from_file is not None:
            self.read_data_file(from_file)
        self._setup()

    def copy(self):
        c = dataset()
        for p in self: c.add_point(p.copy())
        return c
        
    def add_point(self,point):
        self.append(point)
        self._setup()

    def remove_point(self, point):
        self.remove(point)
        self._setup()
        
    def _setup(self):
        self.npoint = len(self)
        self.code = self._gettype("CODE")
        self.target = self._gettype("TARGET")
        self.nugget = self._gettype("NUGGET")
        self.csb = self._gettype("CSB")
        self.other = self._gettype("OTHER")
        self.ncode = np.sum(self.code)
        self.ntarget = np.sum(self.target)
        self.nnugget = np.sum(self.nugget)
        self.ncsb = np.sum(self.csb)
        self.nother = np.sum(self.other)

    def read_data_file(self, filename):
        with open(filename) as f:
            for l in f:
                if l[0]!="#":
                    p = point()
                    p.read_ph_file_line(l)
                    if p.label == "":
                        p.label = "TARGET_%d"%(self.ntarget+1)
                    self.add_point(p)

    def _gettype(self, type):
        return np.array([t.type == type for t in self])

    def getAttribute(self, att, sel = None):
        if sel is None: sel = range(self.npoint)
        elif type(sel[0]) == bool or type(sel[0]) == np.bool_: sel = np.where(sel)[0]
        return np.array([self[p].__dict__[att] for p in sel])

    def toArray(self, sel=None):
        return self.getAttribute("pos",sel=sel)

    def toErrArray(self, sel=None):
        return self.getAttribute("err")

