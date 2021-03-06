
from __future__ import division

from simlib.prof import Distribution
from numpy import *


class RadialDistributionConvolution(object):

    def __init__(self, volume_intersection=None):
        self.cache = {}
        self.volume_intersection = volume_intersection or self.volume_intersection

    def convolude(self, dist, R_particle):
        N = len(dist.data)
        trans_data = (self.get_convolution_matrix(N, dist.prec, R_particle) *
                      dist.data.reshape(N, 1))
        trans_data = array(trans_data).reshape(N)
        trans_data = trans_data[:-int(2*R_particle/dist.prec)]  #cutoff edge effects        
        return dist.__class__(trans_data, dist.prec)

    def get_convolution_matrix(self, N, prec, R_particle):
        key = N,prec,R_particle
        try:
            return self.cache[key]
        except KeyError:
            self.cache[key] = mat = self.make_convolution_matrix(N,prec,R_particle)
            return mat

    def make_convolution_matrix(self, N, prec, R_particle):
        volume_intersection = self.volume_intersection
        base_col = zeros(N, float)
        cols = []
        for i in xrange(N):
            ri = i*prec
            col = base_col.copy()
            cols.append(col)
            for offset in [-1,1]:
                ii = i + (1 if offset==1 else 0)
                while 0<=ii<N:
                    R = ii*prec
                    if R>=R_particle:
                        V_upper = volume_intersection(R+prec, R_particle, ri)
                        V_lower = volume_intersection(R,R_particle, ri)
                        assert V_upper >= V_lower
                        V = V_upper - V_lower
                        if V==0:
                            break
                        col[ii] += V
                    ii += offset
        V_particle = (4/3)*pi*R_particle**3
        return matrix(cols).T / V_particle

    def volume_intersection(self, R, prec):
        raise RuntimeError("volume_intersection, not overriden")

def sphere_f(r, y):
    return pi * (r**2*y - y**3/3)

def volume_integral(r,a,b):
    return sphere_f(r, b) - sphere_f(r, a)

def sphere_volume_intersection(R,r,d):
    assert R>=r
    if d>=r+R:
        return 0
    if d+r<R:
        Vsphere = (4/3)*pi*r**3
        return Vsphere
    yi = (d**2 + r**2 - R**2) / (2*d)
    if yi>0:
        return (volume_integral(R, d-yi, R) +
                volume_integral(r, -r, -yi))
    else:
        Vsphere = (4/3)*pi*r**3
        return Vsphere - (volume_integral(r, -yi, r) -
                          volume_integral(R, d-yi, R))

sphere_convolution = RadialDistributionConvolution(sphere_volume_intersection)

def plane_volume_intersection(R, r, d):
    x = R-d
    if x<-r:
        return 0
    elif x>r:
        return (4/3)*pi*r**3
    else:
        return pi*(r**2*x - x**3/3 + (2/3)*r**3)

plane_convolution = RadialDistributionConvolution(plane_volume_intersection)

class NoConvolution(object):

    @staticmethod
    def convolude(dist):
        return dist

no_convolution = NoConvolution()
