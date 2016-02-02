from __future__ import division

import matplotlib.pyplot as plt
from astropy.table import Table
import numpy as np
from Ska.DBI import DBI
from chandra_aca import star_probs

db = DBI(dbi='sybase', server='sybase', user='aca_read')

stats = db.fetchall('SELECT * from trak_stats_data '
                    'WHERE kalman_datestart > "2014:180" '
                    'AND aoacmag_median is not NULL')

stats = Table(stats)
mags = stats['aoacmag_median']
ok = (mags > 9) & (mags < 11)
stats = stats[ok]
mags = mags[ok]

stats['frac_ms'] = stats['mult_star_samples'] / stats['n_samples']

stats['mag_bin'] = np.round(mags / 0.2) * 0.2
sg = stats.group_by('mag_bin')
sgm = sg.groups.aggregate(np.mean)

plt.figure(1, figsize=(6, 4))
plt.clf()
randx = np.random.uniform(-0.05, 0.05, size=len(stats))
plt.plot(mags + randx, stats['frac_ms'], '.', alpha=0.5,
         label='MS flag rate per obsid')
plt.plot(sgm['mag_bin'], sgm['frac_ms'], 'r', linewidth=5, alpha=0.7,
         label='MS flag rate (0.2 mag bins)')

p_acqs = star_probs.acq_success_prob('2016:001', t_ccd=-15.0, mag=sgm['mag_bin'])
plt.plot(sgm['mag_bin'], 1 - p_acqs, 'g', linewidth=5,
         label='Acq fail rate (model 2016:001, T=-15C)')

plt.legend(loc='upper left', fontsize='small')
plt.xlabel('Magnitude')
plt.title('Acq fail rate compared to MS flag rate')
plt.grid()
plt.tight_layout()
plt.savefig('ms_flag_acq_fails.png')
