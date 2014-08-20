#!/usr/bin/env python

from __future__ import division

import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Column


SOTA2013_FIT = [0.18, 0.99, -0.49,  # Scale
                -1.49, 0.89, 0.28]  # Offset

if 'data' not in globals():
    import cPickle as pickle
    with open('data/mini_acq_table.pkl', 'r') as fh:
        data_all = pickle.load(fh)
        data_all['mag10'] = data_all['mag'] - 10.0
        data_all.sort('year')
        data = data_all.copy()
        data = data[data['year'] > 2007]
        data = data.group_by('quarter')
        data_mean = data.groups.aggregate(np.mean)


def scaled_warm_frac(pars, x, acq_data=None):
    if acq_data is None:
        acq_data = data  # use global data table
    scl0, scl1, scl2 = pars[0:3]
    off0, off1, off2 = pars[3:6]
    m = acq_data['mag10']
    scale = 10**(scl0 + scl1 * m + scl2 * m**2)
    offset = 10**(off0 + off1 * m + off2 * m**2)
    model_fail = offset + scale * (acq_data['warm_pix'] - 0.04)
    return model_fail


def fit_sota_model():
    from sherpa import ui

    data_id = 1
    ui.set_method('simplex')
    ui.set_stat('cash')
    ui.load_user_model(scaled_warm_frac, 'model')
    ui.add_user_pars('model', ['scl0', 'scl1', 'scl2', 'off0', 'off1', 'off2'])
    ui.set_model(data_id, 'model')
    ui.load_arrays(data_id, np.array(data['year']), np.array(data['fail'], dtype=np.float))

    # Initial fit values from SOTA 2013 presentation (modulo typo there)
    start_vals = iter(SOTA2013_FIT)  # Offset
    fmod = ui.get_model_component('model')
    for name in ('scl', 'off'):
        for num in (0, 1, 2):
            comp_name = name + str(num)
            setattr(fmod, comp_name, start_vals.next())
            comp = getattr(fmod, comp_name)
            comp.min = -5
            comp.max = 5
    ui.fit(data_id)
    # conf = ui.get_confidence_results()
    return ui.get_fit_results()


def plot_fit_by_mag(pars, mag0, mag1):
    ok = (data_all['mag'] > mag0) & (data_all['mag'] < mag1)
    data = data_all[ok].group_by('quarter')
    data_mean = data.groups.aggregate(np.mean)

    plt.plot(data_mean['year'], data_mean['fail'], '.b')
    model_fail = Column(scaled_warm_frac(pars, None, data)).group_by(data['quarter'])
    model_fail_mean = model_fail.groups.aggregate(np.mean)
    plt.plot(data_mean['year'], model_fail_mean, '-r')
