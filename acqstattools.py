import numpy as np
import pickle

import scipy.stats as stats
import matplotlib.pyplot as plt

from matplotlib import gridspec

#############################################################################
# 
# Chandra Star Acquisition Prediction functions
#
#############################################################################


def loadacqstats(fname, description=True):
    """
    function: loadacqstats
    inputs  : fname - filename of the acquisition analysis dump
              description - Print Description? Default = True
    returns : outputs from Chandra ACA Analysis
    """
    beta_results = pickle.load(open(fname, 'rb'))
    
    if description:
        print beta_results['description']
    
    return beta_results


class acqPredict8:
    def __init__(self, starcat, betas, summary=True):
        self.starcat = starcat
        self.betas = betas 
        self.starpredictions = []
        for star in starcat:
            self.starpredictions.append(acqPredict1(starcat[star]['mag'], starcat[star]['warm_pix'], betas, agasc=int(star)))
        
        self.pFailMean = np.array([star.pMean for star in self.starpredictions])
        self.pFailLower = np.array([star.pLower for star in self.starpredictions])
        self.pFailUpper = np.array([star.pUpper for star in self.starpredictions])

        self.pSucceedMean = 1 - self.pFailMean
        self.pSucceedUpper = 1 - self.pFailLower
        self.pSucceedLower = 1 - self.pFailUpper
        
        self.exactMean, self.atleastMean = self.pick8(self.pSucceedMean)
        self.exactLower, self.atleastLower = self.pick8(self.pSucceedLower)
        self.exactUpper, self.atleastUpper = self.pick8(self.pSucceedUpper)

        self.exactFail, self.atleastFail = self.pick8(self.pFailMean)
        self.exactFailLower, atleastFailLower = self.pick8(self.pFailLower)
        self.exactFailUpper, atleastUpper = self.pick8(self.pFailUpper)


        self.expectedfailures = np.sum(np.arange(0,9,1)*self.exactFail)

        expectedsquared = np.sum(np.arange(0,9,1)*np.arange(0,9,1)*self.exactFailLower)
        self.variance = expectedsquared - self.expectedfailures**2
        self.stddev = np.sqrt(self.variance)
        if summary:
            print """
Expected Number of Failures: {25:.4} +/- {26}

Star Acquisition Probability Table:
----------------------------------------------
At Least Acquiring [Lower Bound, Upper Bound]:
8 Stars  : {0:<10.8} \t[{1:<10.8}, {2:<10.8}]
7 Stars  : {3:<10.8} \t[{4:<10.8}, {5:<10.8}]
6 Stars  : {6:<10.8} \t[{7:<10.8}, {8:<10.8}]
5 Stars  : {9:<10.8} \t[{10:<10.8}, {11:<10.8}]
4 Stars  : {12:<10.8} \t[{13:<10.8}, {14:<10.8}]
3 Stars  : {15:<10.8} \t[{16:<10.8}, {17:<10.8}]
2 Stars  : {18:<10.8} \t[{19:<10.8}, {20:<10.8}]
1 Stars  : {21:<10.8} \t[{22:<10.8}, {23:<10.8}]

Probability of Acquiring Exactly Zero Stars:
----------------------------------------------
0 Stars  : {24:.5}
            """.format(
            self.atleastMean[8], self.atleastLower[8], self.atleastUpper[8],
            self.atleastMean[7], self.atleastLower[7], self.atleastUpper[7],
            self.atleastMean[6], self.atleastLower[6], self.atleastUpper[6],
            self.atleastMean[5], self.atleastLower[5], self.atleastUpper[5],
            self.atleastMean[4], self.atleastLower[4], self.atleastUpper[4],
            self.atleastMean[3], self.atleastLower[3], self.atleastUpper[3],
            self.atleastMean[2], self.atleastLower[2], self.atleastUpper[2],
            self.atleastMean[1], self.atleastLower[1], self.atleastUpper[1],
            self.atleastMean[0], self.expectedfailures, self.stddev
)

    def pick8(self, probs_success):
        probs = probs_success
        compliment = 1 - probs
        #Probability of getting exactly 8 successes
        p8 = np.prod(probs)
        #Probability of getting exactly 7 successes
        p7 = 0.0
        for i, p in enumerate(probs):
            prob = np.ma.array(probs, mask=False)
            prob.mask[i] = True
            p7 = p7 + prob.prod()*compliment[i]
        #Probability of Getting Exactly 6 Successes
        p6 = 0.0
        for i in np.arange(8):
            for j in np.arange(i+1,8):
                prob = np.ma.array(probs, mask=False)
                prob.mask[i] = True
                prob.mask[j] = True
                p6 = p6 + prob.prod()*compliment[i]*compliment[j]
        #Probability of Getting Exactly 5 Successes
        p5 = 0.0
        for i in np.arange(8):
            for j in np.arange(i+1, 8):
                for k in np.arange(j+1, 8):
                    prob = np.ma.array(probs, mask=False)
                    prob.mask[i] = True
                    prob.mask[j] = True
                    prob.mask[k] = True
                    p5 = p5 + prob.prod()*compliment[i]*compliment[j]*compliment[k]
        #Probability of Getting Exactly 4 Successes
        p4 = 0.0
        for i in np.arange(8):
            for j in np.arange(i+1, 8):
                for k in np.arange(j+1, 8):
                    for l in np.arange(k+1, 8):
                        prob = np.ma.array(probs, mask=False)
                        prob.mask[i] = True
                        prob.mask[j] = True
                        prob.mask[k] = True
                        prob.mask[l] = True
                        p4 = p4 + prob.prod()*compliment[i]*compliment[j]*compliment[k]*compliment[l]
        #Probability of Getting Exactly 3 Successes
        p3 = 0.0
        for i in np.arange(8):
            for j in np.arange(i+1, 8):
                for k in np.arange(j+1, 8):
                    prob = np.ma.array(compliment, mask=False)
                    prob.mask[i] = True
                    prob.mask[j] = True
                    prob.mask[k] = True
                    p3 = p3 + prob.prod()*probs[i]*probs[j]*probs[k]
        #Probability of Getting Exactly 2 Successes
        p2 = 0.0
        for i in np.arange(8):
            for j in np.arange(i+1,8):
                prob = np.ma.array(compliment, mask=False)
                prob.mask[i] = True
                prob.mask[j] = True
                p2 = p2 + prob.prod()*probs[i]*probs[j]
        p1 = 0.0
        for i, p in enumerate(probs):
            prob = np.ma.array(compliment, mask=False)
            prob.mask[i] = True
            p1 = p1 + prob.prod()*probs[i]    
        
        p0 = compliment.prod()
        
        exactly = [p0, p1, p2, p3, p4, p5, p6, p7, p8]
        atleast = [p0, 
                   p8+p7+p6+p5+p4+p3+p2+p1,
                   p8+p7+p6+p5+p4+p3+p2,
                   p8+p7+p6+p5+p4+p3,
                   p8+p7+p6+p5+p4,
                   p8+p7+p6+p5,
                   p8+p7+p6,
                   p8+p7, 
                   p8]
        return exactly, atleast

class acqPredict1:
    def __init__(self, mag, warmpix, beta_results, agasc=""):
        self.agasc = agasc
        self.mag = mag
        self.warmpix = warmpix
        self.pEsts = stats.norm.cdf(np.dot(self.build_matrix_line(self.mag, 
                                        self.warmpix, 
                                        beta_results['centers']['mag'], 
                                        beta_results['centers']['warmpix']), 
                                        beta_results['betamatrix']))
        
        # Calculating Statistics for Probabilities
        self.pMean = np.mean(self.pEsts)
        self.nsims = len(self.pEsts)

        lower_point = np.floor(0.025 * self.nsims).astype(int)
        upper_point = np.ceil(0.975 * self.nsims).astype(int)

        self.pLower = np.sort(self.pEsts)[lower_point]
        self.pUpper = np.sort(self.pEsts)[upper_point]

    def summary(self):
        
        print """
Prediction Summaries for ID {0}:
-----------------------------------------------------------------------------
Star Magnitude                        :  {1}
Warm Pixel Fraction                   :  {2}
Estimated Mean Probability of Failure :  {3}
95% Credible Interval  [Lower, Upper] : [{4}, {5}]

Summary Histogram Plot:""".format(self.agasc, 
                                    self.mag, 
                                    self.warmpix,
                                    self.pMean, 
                                    self.pLower, 
                                    self.pUpper)

        self.histplot()

        print """-----------------------------------------------------------------------------"""

    def histplot(self):
        fig = plt.figure(figsize=(8, 2)) 
        gs = gridspec.GridSpec(1, 2, width_ratios=[1, 3]) 
        ax0 = plt.subplot(gs[0])
        ax0.hist(self.pEsts, normed=True, bins=25)
        plt.yticks(visible=False)
        plt.xticks(rotation=45)
        plt.xlabel("Probability of Failure")

        ax1 = plt.subplot(gs[1])
        ax1.hist(self.pEsts, normed=True, bins=25)
        plt.xlim((0.0,1.0))
        plt.xticks(rotation=45)
        plt.yticks(visible=False)
        plt.xlabel("Probability of Failure")

        plt.show()

    def build_matrix_line(self, m, wp, m_center, wp_center):
        mag = m - m_center
        warm = wp - wp_center
        return np.array([1, mag, mag**2, warm, mag*warm, mag*mag*warm])

#############################################################################
# 
# Chandra Star Acquisition Analysis functions
#
#############################################################################

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