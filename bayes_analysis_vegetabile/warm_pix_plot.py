import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from numpy import genfromtxt
from datetime import datetime as dt
from astropy.time import Time

darkvals = genfromtxt('data/N100.csv', dtype=None, delimiter='\t', names=True)

t = Time(darkvals['Date_YearDOY'])

def scale_offset(mag):
	m = mag - 10.0
	scale = 10**(0.18 + 0.99*m + 0.49*m**2)
	offset = 10**(-1.49 + 0.89*m + 0.28*m**2)
	return scale, offset

def plot_warm_pix(times, n100, mag, fname):
	scale, offset = scale_offset(mag) 
	warm_pix_frac = scale * n100 / 1024**2 + offset
	dates = t.jyear
	F = plt.figure()
	plt.plot(dates, warm_pix_frac, linestyle="-")
	plt.xlabel('Time')
	plt.ylabel('Warm Pix Fraction (%)')
	F.set_size_inches(10,5)
	F.savefig(fname, type='pdf')
	print("Plot:", fname, "... complete")

plot_warm_pix(t, darkvals['N100_es'], 8, 'plots/warm_pix_mag8.pdf')
plot_warm_pix(t, darkvals['N100_es'], 9, 'plots/warm_pix_mag9.pdf')
plot_warm_pix(t, darkvals['N100_es'], 10, 'plots/warm_pix_mag10.pdf')


