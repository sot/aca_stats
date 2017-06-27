import numpy as np
import astropy
import functions as f
import matplotlib.pyplot as plt

from numpy import genfromtxt
from astropy.time import Time

#Loading data from file
acq_data = np.load('data/acq_table.npy')

#Adding fields required for analysis
acq_data = f.add_column(acq_data, 'tstart_jyear' , np.zeros(len(acq_data)))
acq_data = f.add_column(acq_data, 'tstart_quarter' , np.zeros(len(acq_data)))
acq_data = f.add_column(acq_data, 'mag_floor' , np.zeros(len(acq_data)))
acq_data = f.add_column(acq_data, 'year' , np.zeros(len(acq_data)))
acq_data['tstart_jyear'] = Time(acq_data['tstart'], format='cxcsec').jyear
acq_data['year'] = np.floor(acq_data.tstart_jyear)
acq_data['mag_floor'] = np.floor(acq_data['mag'])
# acq_data['tstart_quarter'] = f.quarter_bin(acq['tstart_jyear'])

for acq in acq_data:
    acq.tstart_quarter = f.quarter_bin(acq.tstart_jyear)

#Subsetting the data by the floor of star magnitude
mag8 = f.smlset(acq_data, 'mag_floor', 8.0)
mag9 = f.smlset(acq_data, 'mag_floor', 9.0)
mag10 = f.smlset(acq_data, 'mag_floor', 10.0)

#Subsetting the data by acquisition type, NOID is failed
obs_failed = f.smlset(acq_data, 'obc_id', "NOID")
obs_acq = f.smlset(acq_data, 'obc_id', "ID")

#Subsetting the data by acquisition type & floor of star magnitude
failed9s = f.smlset(mag9, 'obc_id', "NOID")
acqrd9s = f.smlset(mag9, 'obc_id', "ID")
failed10s = f.smlset(mag10, 'obc_id', "NOID")
acqrd10s = f.smlset(mag10, 'obc_id', "ID")


def fails_by_quarter(arr):
    quarters = np.unique(arr['tstart_quarter'])
    obs_counts = []
    failure_counts = []
    for q in quarters:
        obs_inquarter = f.smlset(arr, 'tstart_quarter', q)
        failures = len(f.smlset(obs_inquarter, 'obc_id', 'NOID'))
        counts = len(obs_inquarter)
        failure_counts.append(float(failures))
        obs_counts.append(float(counts))
    failure_rate = np.array(failure_counts) / np.array(obs_counts)
    return [quarters, failure_rate]

#Plottings Figures
def plot_failures(out, fname):
    F = plt.figure()
    plt.plot(out[0],out[1], marker='o', linestyle="")
    plt.xlabel('Quarter')
    plt.ylabel('Acq Fail Rate (%)')
    F.set_size_inches(10,5)
    F.savefig(fname, type='png')
    plt.close()
    print "Plot: {0}... complete".format(fname)

mag8_byquarter = fails_by_quarter(mag8)
mag9_byquarter = fails_by_quarter(mag9)
mag10_byquarter = fails_by_quarter(mag10)

plot_failures(mag8_byquarter, 'plots/mag8_failures_by_quarter.png')
plot_failures(mag9_byquarter, 'plots/mag9_failures_by_quarter.png')
plot_failures(mag10_byquarter, 'plots/mag10_failures_by_quarter.png')