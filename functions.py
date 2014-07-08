import numpy as np
from astropy.time import Time
from datetime import datetime

def add_column(recarray, name, val, index=None):
    #Stolen from Ska.Numpy
    """
    Add a column ``name`` with value ``val`` to ``recarray`` and return a new
    record array.

    :param recarray: Input record array
    :param name: Name of the new column
    :param val: Value of the new column (np.array or list)
    :param index: Add column before index (default: append at end)

    :rtype: New record array with column appended
    """
    if len(val) != len(recarray):
        raise ValueError('Length mismatch: recarray, val = (%d, %d)' % (len(recarray), len(val)))

    arrays = [recarray[x] for x in recarray.dtype.names]
    dtypes = recarray.dtype.descr
    
    if index is None:
        index = len(recarray.dtype.names)

    if not hasattr(val, 'dtype'):
        val = np.array(val)
    valtype = val.dtype.str

    arrays.insert(index, val)
    dtypes.insert(index, (name, valtype))

    return np.rec.fromarrays(arrays, dtype=dtypes)

def quarter_bin(number):
	frac = np.modf(number)[0]
	# print(frac)
	# print(number)
	# print(Time(datetime(2014,1,31)).jyear)
	# Jan31 = np.modf(Time(datetime(2014,1,31)).jyear)[0]
	# Apr30 = np.modf(Time(datetime(2014,4,30)).jyear)[0]
	# Jul31 = np.modf(Time(datetime(2014,7,31)).jyear)[0]
	# Oct31 = np.modf(Time(datetime(2014,10,31)).jyear)[0]

	Jan31 = 0.082135523613942496
	Apr30 = 0.3258042436686992
	Jul31 = 0.57768651608489563
	Oct31 = 0.82956878850109206

	if frac >= 0.0 and frac <= Jan31:
		return np.floor(number)
	elif frac > Jan31 and frac <= Apr30:
		return np.floor(number) + 0.25
	elif frac > Apr30 and frac <= Jul31:
		return np.floor(number) + 0.50
	elif frac > Jul31 and frac <= Oct31:
		return np.floor(number) + 0.75
	else:
		return np.floor(number) + 1

def scale_offset(mag):
	m = mag - 10.0
	scale = 10**(0.18 + 0.99*m + 0.49*m**2)
	offset = 10**(-1.49 + 0.89*m + 0.28*m**2)
	return scale, offset

def subset_by_mag(arr, mag):
	indx_by_mag = np.where(arr.mag_floor==mag)
	return arr[indx_by_mag]

def subset_obcid(subset, val):
	indx_failed = np.where(subset.obc_id==val)
	return subset[indx_failed]



























