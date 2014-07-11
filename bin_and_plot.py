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

failed9s = f.subset_obcid(subset_mag9, "NOID")
acqrd9s = f.subset_obcid(subset_mag9, "ID")

failed10s = f.subset_obcid(subset_mag10, "NOID")
acqrd10s = f.subset_obcid(subset_mag10, "ID")

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
	print "Plot: {0}... complete".format(fname)

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
	print "Plot: {0}... complete".format(fname)

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
	print "Plot: {0}... complete".format(fname)

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
	print "Plot: {0}... complete".format(fname)

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
	print "Plot: {0}... complete".format(fname)

plot_failedheat(obs_failed, 'plots/failedheat.pdf')
plot_failedheat(obs_acq, 'plots/acqheat.pdf')

def histogram_pos(subset, grp, fname):
	F = plt.figure()
	plt.bar(np.array([0,1,2,3,4,5,6,7]), np.bincount(subset[grp]))
	F.set_size_inches(10,5)
	F.savefig(fname, type='pdf')
	plt.close()
	print "Plot: {0}... complete".format(fname)

def hist_stacked(subset1, subset2, grp, fname):
	F = plt.figure()
	plt.bar(np.array([0,1,2,3,4,5,6,7]), np.bincount(subset1[grp]))
	plt.bar(np.array([0,1,2,3,4,5,6,7]), np.bincount(subset2[grp]), bottom=np.bincount(subset1[grp]), color='r')
	F.set_size_inches(10,5)
	F.savefig(fname, type='pdf')
	plt.close()
	print "Plot: {0}... complete".format(fname)


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
	print "Plot: {0}... complete".format(fname)

hist_percents(obs_failed, acq_data, 'cat_pos', 'plots/histogram_catpos_percents.pdf')
hist_percents(obs_failed, acq_data, 'slot', 'plots/histogram_slot_percents.pdf')

hist_percents(f.subset_by_mag(obs_failed, 10.0), subset_mag10, 'cat_pos', 'plots/histogram_catpos_percents_mag10.pdf')
hist_percents(f.subset_by_mag(obs_failed, 10.0), subset_mag10, 'slot', 'plots/histogram_slot_percents_mag10.pdf')

hist_percents(f.subset_by_mag(obs_failed, 9.0), subset_mag9, 'cat_pos', 'plots/histogram_catpos_percents_mag9.pdf')
hist_percents(f.subset_by_mag(obs_failed, 9.0), subset_mag9, 'slot', 'plots/histogram_slot_percents_mag9.pdf')



def boxplots_grp(subset, grp, fname):
	pos0 = f.subset_pos(subset, grp, 0.0).mag
	pos1 = f.subset_pos(subset, grp, 1.0).mag
	pos2 = f.subset_pos(subset, grp, 2.0).mag
	pos3 = f.subset_pos(subset, grp, 3.0).mag
	pos4 = f.subset_pos(subset, grp, 4.0).mag
	pos5 = f.subset_pos(subset, grp, 5.0).mag
	pos6 = f.subset_pos(subset, grp, 6.0).mag
	pos7 = f.subset_pos(subset, grp, 7.0).mag
	all_positions = [pos0, pos1, pos2, pos3, pos4, pos5, pos6, pos7]
	F = plt.figure()
	plt.boxplot(all_positions)
	F.set_size_inches(10,5)
	F.savefig(fname, type='pdf')
	plt.close()
	print "Plot: {0}... complete".format(fname)

boxplots_grp(acq_data, 'slot', 'plots/boxplot_slot_mag.pdf')
boxplots_grp(acq_data, 'cat_pos', 'plots/boxplot_catpos_mag.pdf')
boxplots_grp(obs_failed, 'slot', 'plots/boxplot_slot_mag_failedonly.pdf')
boxplots_grp(obs_failed, 'cat_pos', 'plots/boxplot_catpos_mag_failedonly.pdf')

unique_halfw = np.unique(acq_data.halfw)
hw_strs =  unique_halfw.astype(str)

def plot_halfwidth_failures(subset, fname):
	hw_failrate = []
	hw_acqrate = []
	failed = f.subset_obcid(subset, "NOID")
	acqd = f.subset_obcid(subset, "ID")
	for hw in unique_halfw:
		hw_tot = len(np.where(subset.halfw == hw)[0])
		if hw_tot == 0:
			hw_failrate.append(0.0)
			hw_acqrate.append(0.0)
		else:
			hw_failed = len(np.where(failed.halfw == hw)[0])
			hw_acq = len(np.where(acqd.halfw == hw)[0])
			hw_failrate.append(np.float(hw_failed) / hw_tot)
			hw_acqrate.append(np.float(hw_acq) / hw_tot)
		# print hw, hw_tot, hw_failed, hw_acq

	fig = plt.figure()
	plt.subplot(211)
	plt.bar(np.arange(len(hw_strs)), hw_failrate)
	plt.xticks(np.arange(len(hw_strs)) + 0.5,hw_strs, rotation=90)

	plt.subplot(212)
	plt.bar(np.arange(len(hw_strs)), hw_acqrate)
	plt.xticks(np.arange(len(hw_strs)) + 0.5,hw_strs, rotation=90)

	fig.set_size_inches(13,7)
	fig.savefig(fname, type='pdf')
	print "Plot: {0}... complete".format(fname)

plot_halfwidth_failures(acq_data, 'plots/hw_rates_all.pdf')
plot_halfwidth_failures(subset_mag9, 'plots/hw_rates_mag9.pdf')
plot_halfwidth_failures(subset_mag10, 'plots/hw_rates_mag10.pdf')


#Plots Added 7/11/14

fig = plt.figure()

plt.subplot(321)
plt.plot(obs_failed.mag, obs_failed.warm_pix, marker='.', linestyle='')
plt.subplot(322)
plt.plot(obs_acq.mag, obs_acq.warm_pix, marker='.', linestyle='')
plt.subplot(323)
plt.plot(failed9s.mag, failed9s.warm_pix, marker='.', linestyle='')
plt.subplot(324)
plt.plot(acqrd9s.mag, acqrd9s.warm_pix, marker='.', linestyle='')
plt.subplot(325)
plt.plot(failed10s.mag, failed10s.warm_pix, marker='.', linestyle='')
plt.subplot(326)
plt.plot(acqrd10s.mag, acqrd10s.warm_pix, marker='.', linestyle='')
fig.set_size_inches(15,15)
fig.savefig('plots/scatter_mag_warmpix.png', type='png')
plt.close()
print "Plot: {0}... complete".format('plots/scatter_mag_warmpix.png')

fig = plt.figure()
plt.subplot(211)
plt.plot(obs_failed.mag, obs_failed.warm_pix, marker='.', linestyle='')
plt.subplot(212)
plt.plot(obs_acq.mag, obs_acq.warm_pix, marker='.', linestyle='')
fig.set_size_inches(20,10)
fig.savefig('plots/scatter_mag_warmpixall.png', type='png')
plt.close()
print "Plot: {0}... complete".format('plots/scatter_mag_warmpixall.png')


years = np.unique(acq_data.year)
n_years = len(years)
maxmag = np.max(acq_data.mag)
minmag = np.min(acq_data.mag)
maxwp = np.max(acq_data.warm_pix)
minwp = np.min(acq_data.warm_pix)

fig = plt.figure()
for n, year in enumerate(years, 1):
	left = n * 2 - 1
	right = n * 2
	yeardata = f.subset_pos(acq_data, 'year', year)
	fail_yrd = f.subset_obcid(yeardata, "NOID")

	plt.subplot(n_years, 2, left)
	plt.plot(yeardata.mag, yeardata.warm_pix, marker='.', linestyle='')	
	plt.xlim(minmag, maxmag)
	plt.ylim(minwp, maxwp)

	plt.subplot(n_years, 2, right)
	plt.plot(fail_yrd.mag, fail_yrd.warm_pix, marker='.', linestyle='')
	plt.xlim(minmag, maxmag)
	plt.ylim(minwp, maxwp)
	# print n, year, n_years

fig.set_size_inches(10,25)
fig.savefig('plots/scatter_long.png', type='png')
plt.close()
print "Plot: {0}... complete".format('plots/scatter_long.png')

fig = plt.figure()
plt.hist(obs_failed.mag, bins=20)
fig.set_size_inches(10,5)
fig.savefig('plots/hist_mags.png', type='png')
plt.close()
print "Plot: {0}... complete".format('plots/hist_mags.png')

fig = plt.figure()
for n, year in enumerate(years, 1):
	left = n * 3 - 2
	center = n * 3 - 1
	right = n * 3
	yeardata = f.subset_pos(acq_data, 'year', year)
	fail_yrd = f.subset_obcid(yeardata, "NOID")

	plt.subplot(n_years, 3, left)
	plt.plot(yeardata.mag, yeardata.warm_pix, marker='.', linestyle='')	
	plt.xlim(minmag, maxmag)
	plt.ylim(minwp, maxwp)

	plt.subplot(n_years, 3, center)
	plt.plot(fail_yrd.mag, fail_yrd.warm_pix, marker='.', linestyle='')
	plt.xlim(minmag, maxmag)
	plt.ylim(minwp, maxwp)

	plt.subplot(n_years, 3, right)
	plt.hist(fail_yrd.mag, bins=20)
	plt.xlim(minmag, maxmag)
	
	# print n, year, n_years

fig.set_size_inches(10,25)
fig.savefig('plots/scatter_long_AND_wide.png', type='png')
plt.close()
print "Plot: {0}... complete".format('plots/scatter_long_AND_wide.png')


fig = plt.figure()
mags_by_year = []
pix_by_year = []
for n, year in enumerate(years, 1):
	yr_dat = f.subset_pos(obs_failed, 'year', year)
	mags_by_year.append(yr_dat.mag)
	pix_by_year.append(yr_dat.warm_pix)

F = plt.figure()
plt.boxplot(mags_by_year, vert=False)
plt.yticks(np.arange(len(years))+1, years.astype(int).astype(str))
F.set_size_inches(10,5)
F.savefig("plots/mag_yeardist.png", type='png')
plt.close()
print("Plot: plots/yeardist.png... complete")


F = plt.figure()
plt.boxplot(pix_by_year, vert=False)
plt.yticks(np.arange(len(years))+1, years.astype(int).astype(str))
F.set_size_inches(10,5)
F.savefig("plots/warmpix_yeardist.png", type='png')
plt.close()
print("Plot: plots/yeardist.png... complete")

