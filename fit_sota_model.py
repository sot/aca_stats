#!/usr/bin/env python

from __future__ import division

import numpy as np


SOTA2013_FIT = [0.18, 0.99, -0.49,  # Scale
                -1.49, 0.89, 0.28]  # Offset

if 'data' not in globals():
    import cPickle as pickle
    with open('data/mini_acq_table.pkl', 'r') as fh:
        data = pickle.load(fh)
        data['mag10'] = data['mag'] - 10.0
        data = data[data['year'] > 2007]
        data = data.group_by('quarter')
        data_mean = data.groups.aggregate(np.mean)


def scaled_warm_frac(pars, x):
    scl0, scl1, scl2 = pars[0:3]
    off0, off1, off2 = pars[3:6]
    m = data['mag10']
    scale = 10**(scl0 + scl1 * m + scl2 * m**2)
    offset = 10**(off0 + off1 * m + off2 * m**2)
    model_y = offset + scale * data['warm_pix']
    return model_y


def fit_sota_model():
    from sherpa import ui

    data_id = 1
    ui.set_method('simplex')
    ui.set_stat('cash')
    ui.load_user_model(scaled_warm_frac, 'model')
    ui.add_user_pars('model', ['scl0', 'scl1', 'scl2', 'off0', 'off1', 'off2'])
    ui.set_model(data_id, 'model')
    ui.load_arrays(data_id, data['year'], data['fail'].astype(np.float))

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
