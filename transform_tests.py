import mirrorfit as mf
import theano.tensor as tt
import theano
import pymc3 as pm
import numpy as np
import matplotlib.pyplot as plt
import unittest
import sys

def build_trace(fName = '20181205_primary_receiver.txt'):
        ds1 = mf.dataset(from_file=fName) 
        np.random.seed(1)

        with pm.Model() as model:
            mean_scale = 50.
            sd_scale_pos=1000.
            sd_scale_angle=50.
            d1t = ds1.to_tensors()
            tx1=pm.Normal('tx1', mu=np.random.normal()*mean_scale, sd=sd_scale_pos) 
            ty1=pm.Normal('ty1', mu=np.random.normal()*mean_scale, sd=sd_scale_pos) 
            tz1=pm.Normal('tz1', mu=np.random.normal()*mean_scale, sd=sd_scale_pos) 
            rx1=pm.Normal('rx1', mu=np.random.normal()*mean_scale, sd=sd_scale_pos) 
            ry1=pm.Normal('ry1', mu=np.random.normal()*mean_scale, sd=sd_scale_pos) 
            rz1=pm.Normal('rz1', mu=np.random.normal()*mean_scale, sd=sd_scale_pos) 
            s1=pm.Normal('s1', mu=100., sd=10.) 
            t1 = mf.TheanoTransform({'tx':tx1, 'ty':ty1, 'tz':tz1, 'rx':rx1, 'ry':ry1, 'rz':rz1, 's':s1})

            tx2=pm.Normal('tx2', mu=np.random.normal()*mean_scale, sd=sd_scale_angle) 
            ty2=pm.Normal('ty2', mu=np.random.normal()*mean_scale, sd=sd_scale_angle) 
            tz2=pm.Normal('tz2', mu=np.random.normal()*mean_scale, sd=sd_scale_angle) 
            rx2=pm.Normal('rx2', mu=np.random.normal()*mean_scale, sd=sd_scale_angle) 
            ry2=pm.Normal('ry2', mu=np.random.normal()*mean_scale, sd=sd_scale_angle) 
            rz2=pm.Normal('rz2', mu=np.random.normal()*mean_scale, sd=sd_scale_angle) 
            s2=pm.Normal('s2', mu=100., sd=10.) 
            t2 = mf.TheanoTransform({'tx':tx2, 'ty':ty2, 'tz':tz2, 'rx':rx2, 'ry':ry2, 'rz':rz2, 's':s2})

            #check associativity
            d2 = t1*d1t
            d3 = t2*d2
            d3p = (t2*t1)*d1t
            diffassc = d3.pos-d3p.pos 
            distassc = diffassc.norm(L=2, axis=0)
            differrassc = d3.err-d3p.err 
            disterrassc = differrassc.norm(L=2, axis=0)
            pm.Deterministic('associativity_position', (distassc).max())
            pm.Deterministic('associativity_error', (disterrassc).max())

            #check invertability
            t1i = ~t1
            dsuccessive = t1*d1t
            dsuccessive = t1i*dsuccessive

            dsingle = (t1*t1i)*d1t
            diffinvsuc = d1t.pos-dsuccessive.pos
            diffinvsucerr = dsuccessive.err-dsingle.err
            diffinvsin = d1t.pos-dsuccessive.pos
            diffinvsinerr = dsuccessive.err-dsingle.err
            pm.Deterministic('invertability_position_successive', (diffinvsuc).max())
            pm.Deterministic('invertability_error_successive', (diffinvsucerr).max())
            pm.Deterministic('invertability_position_single', (diffinvsin).max())
            pm.Deterministic('invertability_error_single', (diffinvsinerr).max())

        with model:
            trace = pm.sample(500, tune=500)
        return trace, ds1

class TestTransforms(unittest.TestCase):
    
    @classmethod
    def setUpClass(self):
        self.trace, self.dataset = build_trace()

    def test_associative_position(self):
        self.assertAlmostEqual(first=np.max(self.trace.get_values('associativity_position')),second=0.)
            
    def test_associative_error(self):
        self.assertAlmostEqual(first=np.max(self.trace.get_values('associativity_error')),second=0.)

    def test_invertability_position_successive(self):
        self.assertAlmostEqual(first=np.max(self.trace.get_values('invertability_position_successive')),second=0.)

    def test_invertability_position_single(self):
        self.assertAlmostEqual(first=np.max(self.trace.get_values('invertability_position_single')),second=0.)

    def test_invertability_error_successive(self):
        self.assertAlmostEqual(first=np.max(self.trace.get_values('invertability_error_successive')),second=0.)

    def test_invertability_error_single(self):
        self.assertAlmostEqual(first=np.max(self.trace.get_values('invertability_error_single')),second=0.)

def suite():
    suite = unittest.TestSuite()
    suite.addTest(TestTransforms('test_associative_position'))
    suite.addTest(TestTransforms('test_associative_error'))
    suite.addTest(TestTransforms('test_invertability_position_successive'))
    suite.addTest(TestTransforms('test_invertability_position_single'))
    suite.addTest(TestTransforms('test_invertability_error_successive'))
    suite.addTest(TestTransforms('test_invertability_error_single'))
    return suite

if __name__ == '__main__':

    suite = suite()
    ret = not unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful()

    sys.exit(ret)
