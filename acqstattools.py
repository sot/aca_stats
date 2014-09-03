import numpy as np

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

def subset_range_tstart_jyear(dset, start, end):
    return dset[(dset['tstart_jyear']>=start) & (dset['tstart_jyear']<=end)]

def subset_range_warmpix(dset, wpf):
    return dset[(dset['warm_pix']>=(wpf - 0.01)) & (dset['warm_pix']<=(wpf + 0.01))]