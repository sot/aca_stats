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

for acq in acq_data:
	acq.tstart_quarter = f.quarter_bin(acq.tstart_jyear)
	acq.mag_floor = np.floor(acq.mag)

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

subset_mag9 = f.subset_by_mag(acq_data, 9.0)
subset_mag10 = f.subset_by_mag(acq_data, 10.0)

obs_failed = f.subset_obcid(acq_data, "NOID")
obs_acq = f.subset_obcid(acq_data, "ID")

darkvals = genfromtxt('data/N100.csv', dtype=None, delimiter='\t', names=True)

t = Time(darkvals['Date_YearDOY'])

#Plottings Figures
def plot_failures(out, fname):
	F = plt.figure()
	plt.plot(out[0],out[1], marker='o', linestyle="")
	plt.xlabel('Quarter')
	plt.ylabel('Acq Fail Rate (%)')
	F.set_size_inches(10,5)
	F.savefig(fname, type='pdf')
	plt.close()
	print("Plot:", fname, "... complete")

plot_failures(mag8, 'plots/mag8.pdf')
plot_failures(mag9, 'plots/mag9.pdf')
plot_failures(mag10, 'plots/mag10.pdf')

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
	plt.close()
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
	plt.close()
	print("Plot:", fname, "... complete")

plot_overlay(acq_data, 10.0, t, darkvals['N100_es'], 'plots/overlay_mag10.pdf')	
plot_overlay(acq_data, 9.0, t, darkvals['N100_es'], 'plots/overlay_mag9.pdf')

def plot_warmpix_fromfile(subset, fname):
	F = plt.figure()
	plt.plot(subset.tstart_jyear,subset.warm_pix, marker='o', linestyle="")
	plt.xlabel('Time')
	plt.ylabel('Warm Pixel Fraction (%)')
	F.set_size_inches(10,5)
	F.savefig(fname, type='pdf')
	plt.close()
	print("Plot:", fname, "... complete")

plot_warmpix_fromfile(subset_mag9, 'plots/subsetmag9.pdf')
plot_warmpix_fromfile(subset_mag10, 'plots/subsetmag10.pdf')


def plot_failedheat(subset, fname):
	F = plt.figure()
	heatmap, xedges, yedges = np.histogram2d(subset.yang,subset.zang, bins=100)
	extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
	plt.imshow(heatmap, extent=extent)
	plt.colorbar()
	F.set_size_inches(5,5)
	F.savefig(fname, type='pdf')
	plt.close()
	print("Plot:", fname, "... complete")

plot_failedheat(obs_failed, 'plots/failedheat.pdf')
plot_failedheat(obs_acq, 'plots/acqheat.pdf')

def histogram_pos(subset, grp, fname):
	F = plt.figure()
	plt.bar(np.array([0,1,2,3,4,5,6,7]), np.bincount(subset[grp]))
	F.set_size_inches(10,5)
	F.savefig(fname, type='pdf')
	plt.close()
	print("Plot:", fname, "... complete")

def hist_stacked(subset1, subset2, grp, fname):
	F = plt.figure()
	plt.bar(np.array([0,1,2,3,4,5,6,7]), np.bincount(subset1[grp]))
	plt.bar(np.array([0,1,2,3,4,5,6,7]), np.bincount(subset2[grp]), bottom=np.bincount(subset1[grp]), color='r')
	F.set_size_inches(10,5)
	F.savefig(fname, type='pdf')
	plt.close()
	print("Plot:", fname, "... complete")


hist_stacked(obs_acq, obs_failed, 'slot', 'plots/stacked_slots.pdf')
hist_stacked(obs_acq, obs_failed, 'cat_pos', 'plots/stacked_cat_pos.pdf')

hist_stacked(f.subset_by_mag(obs_acq,9.0), f.subset_by_mag(obs_failed,9.0), 'slot', 'plots/stacked_slots_mag9.pdf')
hist_stacked(f.subset_by_mag(obs_acq,9.0), f.subset_by_mag(obs_failed,9.0), 'cat_pos', 'plots/stacked_cat_pos_mag9.pdf')

hist_stacked(f.subset_by_mag(obs_acq,10.0), f.subset_by_mag(obs_failed,10.0), 'slot', 'plots/stacked_slots_mag10.pdf')
hist_stacked(f.subset_by_mag(obs_acq,10.0), f.subset_by_mag(obs_failed,10.0), 'cat_pos', 'plots/stacked_cat_pos_mag10.pdf')

def hist_percents(subset, lrgset, grp, fname):
	F = plt.figure()
	totals = np.bincount(lrgset[grp]).astype(float)
	parts = np.bincount(subset[grp]).astype(float)
	percs = parts / totals
	plt.bar(np.array([0,1,2,3,4,5,6,7]), percs)
	F.set_size_inches(10,5)
	F.savefig(fname, type='pdf')
	plt.close()
	print("Plot:", fname, "... complete")

hist_percents(obs_failed, acq_data, 'cat_pos', 'plots/histogram_catpos_percents.pdf')
hist_percents(obs_failed, acq_data, 'slot', 'plots/histogram_slot_percents.pdf')

hist_percents(f.subset_by_mag(obs_failed, 10.0), subset_mag10, 'cat_pos', 'plots/histogram_catpos_percents_mag10.pdf')
hist_percents(f.subset_by_mag(obs_failed, 10.0), subset_mag10, 'slot', 'plots/histogram_slot_percents_mag10.pdf')

hist_percents(f.subset_by_mag(obs_failed, 9.0), subset_mag9, 'cat_pos', 'plots/histogram_catpos_percents_mag9.pdf')
hist_percents(f.subset_by_mag(obs_failed, 9.0), subset_mag9, 'slot', 'plots/histogram_slot_percents_mag9.pdf')

def subset_pos(subset, grp, val):
	indx = np.where(subset[grp]==val)
	return subset[indx]

def boxplots_grp(subset, grp, fname):
	pos0 = subset_pos(subset, grp, 0.0).mag
	pos1 = subset_pos(subset, grp, 1.0).mag
	pos2 = subset_pos(subset, grp, 2.0).mag
	pos3 = subset_pos(subset, grp, 3.0).mag
	pos4 = subset_pos(subset, grp, 4.0).mag
	pos5 = subset_pos(subset, grp, 5.0).mag
	pos6 = subset_pos(subset, grp, 6.0).mag
	pos7 = subset_pos(subset, grp, 7.0).mag
	all_positions = [pos0, pos1, pos2, pos3, pos4, pos5, pos6, pos7]
	F = plt.figure()
	plt.boxplot(all_positions)
	F.set_size_inches(10,5)
	F.savefig(fname, type='pdf')
	plt.close()
	print("Plot:", fname, "... complete")

boxplots_grp(acq_data, 'slot', 'plots/boxplot_slot_mag.pdf')
boxplots_grp(acq_data, 'cat_pos', 'plots/boxplot_catpos_mag.pdf')
boxplots_grp(obs_failed, 'slot', 'plots/boxplot_slot_mag_failedonly.pdf')
boxplots_grp(obs_failed, 'cat_pos', 'plots/boxplot_catpos_mag_failedonly.pdf')


