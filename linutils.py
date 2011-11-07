'''Assorted linear algebra utilities
'''

from __future__ import division
from __future__ import absolute_import

import transformations
import numpy as np

from .util import all_epsilon_eq, epsilon_eq

vx,vy,vz = cartesian_basis = np.identity(3)
identity_rotation = np.matrix(np.identity(3))
theta_epsilon = 1e-8 * np.pi

def vlen(v): return np.sqrt((v**2).sum())

def vnorm(v): return v / vlen(v)

def vperp(r, orient):
    d = np.dot(r, orient)
    return vnorm((orient - d * r)) if abs(d-1) > 1e-9 else orient

def canonical_norm(_v):
    v = vnorm(_v)
    if np.isnan(v).any():
        return vx
    for basis_vector in cartesian_basis:
        if abs(np.dot(v, basis_vector)) < (1 - 1e-6):
            return vnorm(np.cross(v, basis_vector))
    raise ValueError("couldn't find a canonical normal for %r" % (_v,))

def to_rotation_matrix(op):
    if op is None: #shorthand for identity
        op = identity_rotation.copy()
    else:
        op = np.matrix(op)
        if op.shape == (1,4): #quaternion
            op = transformations.quaternion_matrix(np.array(op)[0, ...])
        elif op.shape == (1,3): #euler angles
            op = transformations.euler_matrix(*np.array(op).reshape(3))
        if op.shape == (4,4): #universal transformation matrix
            #ensure all transformations are rotational
            def check_slice(slc):
                assert all_epsilon_eq(slc, 0, 1e-10), 'bad slice %r' % (slc,)
            check_slice(op[-1, :3:])
            check_slice(op[:3:, -1])
            assert epsilon_eq(op[3,3], 1, 1e-10)
            op = np.asmatrix(op[:3, :3])
    assert op.shape == (3,3)
    return op

def calculate_euler_matrix(angles):
    return np.matrix(transformations.euler_matrix(*angles)[:3:, :3:])

def calculate_quaternion_matrix(angle, direction):
    return np.matrix(transformations.rotation_matrix(angle, direction)[:3:, :3:])

def times_column3(mat, vec):
    return np.array(np.asmatrix(mat) * np.asarray(vec).reshape(3,1)).reshape(3)

def rotate_positions(mat, positions):
    return np.array(np.asmatrix(mat) * np.asarray(positions).T).T.copy()

def canonical_rot_mat(start_on_unit_sphere, end_on_unit_sphere):
    theta = np.arccos(np.dot(start_on_unit_sphere, end_on_unit_sphere))
    if epsilon_eq(theta, 0, theta_epsilon):
        mat = identity_rotation.copy()
    elif epsilon_eq(theta, np.pi, theta_epsilon):
        mat = identity_rotation.copy()
        mat[2,2] = -1.0
    else:
        v = np.cross(start_on_unit_sphere, end_on_unit_sphere)
        mat = np.matrix(transformations.rotation_matrix(theta, v)[:3,:3])
    end_test = times_column3(mat, start_on_unit_sphere)
    assert sum(abs(end_on_unit_sphere - end_test)) < 1e-2, '%s -> %s' % (end_on_unit_sphere, end_test)
    return mat

def colinear(a, b):
    return epsilon_eq(abs(np.dot(a, b)), 1, 1e-6)

def signed_angle_between_unit_vecs(va, vb, unit_axis):
    d = np.dot(va, vb)
    if abs(1.0 - d) < 1e-6:
        return 0
    elif abs(-1.0 - d) < 1e-6:
        return np.pi
    if colinear(va, unit_axis) or colinear(vb, unit_axis):
        return 0
    theta = np.arccos(d)
    if abs(theta - np.pi) < theta_epsilon:
        return np.pi
    dd = np.dot(vnorm(np.cross(va,vb)), unit_axis)
    assert abs(abs(dd) - 1.0) < 1e-4, 'bad axis %s for angle between %s and %s' % (unit_axis, va, vb)
    if dd<0: theta = -theta
    assert -np.pi<=theta<=np.pi
    return theta

def distance_from_line(x1, x2, point):
    return vlen(np.cross(point-x1, point-x2)) / vlen(x2 - x1)
