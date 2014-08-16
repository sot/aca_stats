import cPickle as pickle

import numpy as np
from astropy.table import Table
from astropy.time import Time

year_q0 = 1999.0 + 31. / 365.25  # Jan 31 approximately

acqs = Table(np.load('data/acq_table.npy'))
acqs = acqs['tstart', 'obsid', 'obc_id', 'warm_pix', 'mag']
year = (Time(acqs['tstart'], format='cxcsec').jyear).astype('f4')
quarter = (np.trunc((year - 1999.0) * 4)).astype('f4')
fail = (acqs['obc_id'] != 'ID').astype(np.int)
warm_pix = acqs['warm_pix'].astype('f4')
mag = acqs['mag'].astype('f4')

t = Table([year, quarter, warm_pix, mag, fail],
          names=['year', 'quarter', 'warm_pix', 'mag', 'fail'])

with open('data/mini_acq_table.pkl', 'w') as fh:
    pickle.dump(t, fh, protocol=-1)
