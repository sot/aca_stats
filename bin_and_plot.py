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
acq_data['tstart_jyear'] = Time(acq_data['tstart'], format='cxcsec').jyear

count = 0
for acq in acq_data:
	count += 1
	acq.tstart_quarter = f.quarter_bin(acq.tstart_jyear)
	# print acq.tstart_jyear, f.quarter_bin(acq.tstart_jyear)
	acq.mag_floor = np.floor(acq.mag)
	# print acq.mag_floor, acq.mag

#Function to calculate Acquistions by Quarter
def acq_byquarter(arr, mag=None):
	quarters = np.unique(arr.tstart_quarter)
	obs_counts = []
	failure_counts = []
	if mag is None:
		for q in quarters:
			failures = len(np.where((arr.tstart_quarter == q) & (arr.obc_id == "NOID"))[0])
			counts = len(np.where(arr.tstart_quarter == q)[0])
			failure_counts.append(float(failures))
			obs_counts.append(float(counts))
	else:
		for q in quarters:
			failures = len(np.where((arr.tstart_quarter == q) & (arr.obc_id == "NOID") & (arr.mag_floor == mag))[0])
			counts = len(np.where((arr.tstart_quarter == q) & (arr.mag_floor == mag))[0])
			failure_counts.append(float(failures))
			obs_counts.append(float(counts))
	failure_rate = np.array(failure_counts) / np.array(obs_counts)
	return [quarters, failure_rate]

mag8 = acq_byquarter(acq_data, mag=8.0)
mag9 = acq_byquarter(acq_data, mag=9.0)
mag10 = acq_byquarter(acq_data, mag=10.0)


#Plottings Figures
def plot_failures(out, fname):
	F = plt.figure()
	plt.plot(out[0],out[1], marker='o', linestyle="")
	plt.xlabel('Quarter')
	plt.ylabel('Acq Fail Rate (%)')
	F.set_size_inches(10,5)
	F.savefig(fname, type='pdf')
	print("Plot:", fname, "... complete")

plot_failures(mag8, 'plots/mag8.pdf')
plot_failures(mag9, 'plots/mag9.pdf')
plot_failures(mag10, 'plots/mag10.pdf')

darkvals = genfromtxt('data/N100.csv', dtype=None, delimiter='\t', names=True)

t = Time(darkvals['Date_YearDOY'])

def plot_warm_pix(times, n100, mag, fname):
	scale, offset = f.scale_offset(mag) 
	warm_pix_frac = scale * n100 / 1024**2 + offset
	dates = t.jyear
	F = plt.figure()
	plt.plot(dates, warm_pix_frac, linestyle="-", color='r')
	plt.xlabel('Time')
	plt.ylabel('Warm Pix Fraction (%)')
	F.set_size_inches(10,5)
	F.savefig(fname, type='pdf')
	print("Plot:", fname, "... complete")

plot_warm_pix(t, darkvals['N100_es'], 8, 'plots/warm_pix_mag8.pdf')
plot_warm_pix(t, darkvals['N100_es'], 9, 'plots/warm_pix_mag9.pdf')
plot_warm_pix(t, darkvals['N100_es'], 10, 'plots/warm_pix_mag10.pdf')


def plot_overlay(acq_data, mag, times, n100, fname):
	magData = acq_byquarter(acq_data, mag=mag)
	scale, offset = f.scale_offset(mag) 
	warm_pix_frac = scale * n100 / 1024**2 + offset
	dates = t.jyear
	F = plt.figure()
	plt.plot(magData[0], magData[1], marker='o', linestyle="")
	plt.plot(dates, warm_pix_frac, linestyle="-", color='r')
	F.set_size_inches(10,5)
	F.savefig(fname, type='pdf')
	print("Plot:", fname, "... complete")

plot_overlay(acq_data, 10.0, t, darkvals['N100_es'], 'plots/overlay_mag10.pdf')	
plot_overlay(acq_data, 9.0, t, darkvals['N100_es'], 'plots/overlay_mag9.pdf')	


