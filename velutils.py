'''Velocity utilities
'''

from __future__ import division
from __future__ import absolute_import

from itertools import izip

import numpy as np

from .linutils import cartesian_basis, vlen, vnorm


def calculate_collective_translational_velocity(velocities):
    return np.asarray(velocities).mean(axis=0)

def calculate_collective_angular_velocity(positions, velocities, translational_velocities=None):

    positions = np.asarray(positions)
    velocities = np.asarray(velocities)

    com = positions.mean(axis=0)

    if translational_velocities is None:
        translational_velocities = calculate_collective_translation_velocity(velocities)

    angular_velocities = np.array(list(calculate_angular_velocity_contribution(position, velocity,
                                                                               com, translational_velocity)
                                       for position, velocity in izip(positions, velocities)))
    return angular_velocities.mean(axis=0)

def calculate_angular_velocity_contribution(position, space_velocity, com, com_velocity):
    internal_velocity = space_velocity - com_velocity
    r = position - com
    radius = vlen(r)

    r_n = vnorm(r)
    v_trans = r_n * np.dot(r_n, internal_velocity)
    v_rot = internal_velocity - v_trans

    omega = np.array(list(calculate_angular_velocity(v_rot, axis, radius)
                      for axis in cartesian_basis))
    return omega

def calculate_angular_velocity(v_r, axis, radius):
    v = np.cross(v_r, axis).sum()
    return v / radius

