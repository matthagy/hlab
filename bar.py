'''Bennet Acceptance Ratio method
'''

from __future__ import division

import numpy as np

from .binsolve import binsolve
from .uncertain import UncertainNumber


def solve_free_energy(sample_A, sample_B, u_A, u_B, C_min=-100, C_max=100, **kwds):
    ''' sample_A - sequences of configurations sampled with potetntial A
        sample_B - sequences of configurations sampled with potetntial B
        u_A - evaluates energies for sequences of configurations (sample_x) using potential A
        u_B - evaluates energies for sequences of configurations (sample_x) using potential B
        C_min, C_max - limits for binary solving of free energy
        f - function satisfying f(x)/f(-x) = e^-x
    '''
    Us_A = u_B(sample_A) - u_A(sample_A)
    Us_B = u_A(sample_B) - u_B(sample_B)

    n_F = len(Us_A)
    n_R = len(Us_B)

    def f(x):
        return 1.0 / (1.0 + np.exp(x))

    def test_C(C):
        return np.log(f(Us_A - C).mean() / f(Us_B + C).mean())

    return binsolve(test_C, 0.0, C_min, C_max, **kwds)

def fermi_diarc(x):
    return 1.0 / (1.0 + np.exp(x))

def calculate_variance(sample_A, sample_B, u_A, u_B, free_energy):
    work_F = u_B(sample_A) - u_A(sample_A)
    work_R = u_A(sample_B) - u_B(sample_B)
    n_F = len(work_F)
    n_R = len(work_R)
    n_t = n_F + n_R
    M = np.log(n_F/n_R)
    works = np.concatenate([work_F, work_R])
    return 1/n_t * (np.mean(1.0 / (2.0 + 2.0 * np.cosh(M + works - free_energy)))**-1 -
                    (n_t / n_F + n_t / n_R))

def solve_free_energy_uncertain(sample_A, sample_B, u_A, u_B, **kwds):
    free_energy = solve_free_energy(sample_A, sample_B, u_A, u_B, **kwds)
    variance = calculate_variance(sample_A, sample_B, u_A, u_B, free_energy)
    return UncertainNumber(free_energy, variance**0.5)
