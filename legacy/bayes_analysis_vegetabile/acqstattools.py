import numpy as np
import pickle
import sys

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
    inputs  : fname - filename of the acquisition analysis dump
              description - Print Description? Default = True
    returns : outputs from Chandra ACA Analysis
    """
    beta_results = pickle.load(open(fname, 'rb'))
    
    if description:
        print beta_results['description']
    
    return beta_results


class acqPredictCatalog:
    """
    Class for a estimated probability of the failure for an entire
    star catalog.
    """
    def __init__(self, starcat, betas, summary=True):
        """
        inputs: starcat - dictionary of a specific star catalog.
                betas - dictionary of results loaded by the function
                        loadacqstats.  Results from Chandra ACA Analysis
                summary - If true, prints a summary table of Probability
                          estimates and the associated pdfof failure.  
                          Defaults True.

        returns: acqPredict8 object.

        Available Methods:
        - pick8: estimates the probability of 8 stars.

        Available Attributes:
        - starcat: inputted star catalog; type: dictionary
        - betas: inputed beta results; type: dictionary
        - starpredictions: Individual predictions for each star in the
                           the catalog.  Each prediction is an object 
                           of type acqPredict1; type: list
        - 
        """
        # Storing inputs
        self.starcat = starcat
        self.betas = betas 

        # Building a list of acqPredict1 objects.  
        self.starpredictions = []
        for star in starcat:
            self.starpredictions.append(acqPredict1(starcat[star]['mag'], starcat[star]['warm_pix'], betas, agasc=int(star)))
        
        # Extracting Probabilities for each star in the catalog
        self.pFailMean = np.array([star.pMean for star in self.starpredictions])
        self.pFailLower = np.array([star.pLower for star in self.starpredictions])
        self.pFailUpper = np.array([star.pUpper for star in self.starpredictions])

        # Calculating the Individual Probabilities of the Success
        self.pSucceedMean = 1 - self.pFailMean
        self.pSucceedUpper = 1 - self.pFailLower
        self.pSucceedLower = 1 - self.pFailUpper
        

        # Calculating Probabilities for the groups of stars
        if len(self.pFailMean) == 8:
            # Calculating Using Success Probabilities
            self.exactSucceedMean, self.atleastSucceedMean, self.atmostSucceeedMean = self.pick8(self.pSucceedMean)
            # Interval Estimates for Success
            self.exactSucceedLower, self.atleastSucceedLower, self.atmostSucceedLower = self.pick8(self.pSucceedLower)
            self.exactSucceedUpper, self.atleastSucceedUpper, self.atmostSucceedLower = self.pick8(self.pSucceedUpper)

            #Calculating Using Failure Probabilities
            self.exactFailMean, self.atleastFailMean, self.atmostFailMean = self.pick8(self.pFailMean)
            
            # Interval Estimates for Failures
            self.exactFailLower, self.atleastFailLower, self.atmostFailLower = self.pick8(self.pFailLower)
            self.exactFailUpper, self.atleastFailUpper, self.atmostFailUpper = self.pick8(self.pFailUpper)
            
            # Calculating Summary Statistics
            self.expectedfailures = np.sum(np.arange(0,9,1)*self.exactFailMean)

            expectedsquared = np.sum(np.arange(0,9,1)*np.arange(0,9,1)*self.exactFailLower)
            self.variance = expectedsquared - self.expectedfailures**2
            self.stddev = np.sqrt(self.variance)


            if summary:
                print """
Expected Number of Failures: {0:.4} +/- {1}""".format(self.expectedfailures, 
                    self.stddev)

                print """
Star Acquisition Probability Tables:
------------------------------------------------------------------

At Least Acquiring, At Most Failing [Lower Bound, Upper Bound]:
------------------------------------------------------------------
8 Stars Acquire, 0 failing: {0:<10.8} \t[{9:<10.8}, {18:<10.8}]
7 Stars Acquire, 1 failing: {1:<10.8} \t[{10:<10.8}, {19:<10.8}]
6 Stars Acquire, 2 failing: {2:<10.8} \t[{11:<10.8}, {20:<10.8}]
5 Stars Acquire, 3 failing: {3:<10.8} \t[{12:<10.8}, {21:<10.8}]
4 Stars Acquire, 4 failing: {4:<10.8} \t[{13:<10.8}, {22:<10.8}]
3 Stars Acquire, 5 failing: {5:<10.8} \t[{14:<10.8}, {23:<10.8}]
2 Stars Acquire, 6 failing: {6:<10.8} \t[{15:<10.8}, {24:<10.8}]
1 Stars Acquire, 7 failing: {7:<10.8} \t[{16:<10.8}, {25:<10.8}]
0 Stars Acquire, 8 failing: {8:<10.8} \t[{17:<10.8}, {26:<10.8}]
                """.format(self.atleastSucceedMean[8],
                    self.atleastSucceedMean[7],
                    self.atleastSucceedMean[6],
                    self.atleastSucceedMean[5],
                    self.atleastSucceedMean[4],
                    self.atleastSucceedMean[3],
                    self.atleastSucceedMean[2],
                    self.atleastSucceedMean[1],
                    self.atleastSucceedMean[0],

                    self.atleastSucceedLower[8],
                    self.atleastSucceedLower[7],
                    self.atleastSucceedLower[6],
                    self.atleastSucceedLower[5],
                    self.atleastSucceedLower[4],
                    self.atleastSucceedLower[3],
                    self.atleastSucceedLower[2],
                    self.atleastSucceedLower[1],
                    self.atleastSucceedLower[0],

                    self.atleastSucceedUpper[8],
                    self.atleastSucceedUpper[7],
                    self.atleastSucceedUpper[6],
                    self.atleastSucceedUpper[5],
                    self.atleastSucceedUpper[4],
                    self.atleastSucceedUpper[3],
                    self.atleastSucceedUpper[2],
                    self.atleastSucceedUpper[1],
                    self.atleastSucceedUpper[0])

                print"""
    Probability Distribution Function for Star Catalog
                """
                F = plt.figure()
                plt.bar(np.arange(0,9,1), self.exactFailMean, width=.75)
                plt.xlabel('Exactly Fail')
                plt.ylabel('Probability')
                F.set_size_inches(8,2)
                plt.title('Probability Distribution Function for Star Catalog')
                plt.show()
        
        else:
            print """Star Catalogs need 8 Stars...

...or so I've been told.
"""


    def pick8(self, probs_success):
        """ 
        Given 8 stars and their individual probabilities, calculate the 
        expected probabilites for the entire group of 8.
        """
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
        
        #Probability of Getting Exactly 1 Successes
        p1 = 0.0
        for i, p in enumerate(probs):
            prob = np.ma.array(compliment, mask=False)
            prob.mask[i] = True
            p1 = p1 + prob.prod()*probs[i]    
        
        ##Probability of Getting Exactly 0 Successes
        p0 = compliment.prod()
        
        exactly = [p0, p1, p2, p3, p4, p5, p6, p7, p8]
        atleast = [p8+p7+p6+p5+p4+p3+p2+p1+p0,
                   p8+p7+p6+p5+p4+p3+p2+p1,
                   p8+p7+p6+p5+p4+p3+p2,
                   p8+p7+p6+p5+p4+p3,
                   p8+p7+p6+p5+p4,
                   p8+p7+p6+p5,
                   p8+p7+p6,
                   p8+p7, 
                   p8]
        atmost = [p0,
                  p0+p1,
                  p0+p1+p2,
                  p0+p1+p2+p3,
                  p0+p1+p2+p3+p4,
                  p0+p1+p2+p3+p4+p5,
                  p0+p1+p2+p3+p4+p5+p6,
                  p0+p1+p2+p3+p4+p5+p6+p7,
                  p0+p1+p2+p3+p4+p5+p6+p7+p8]
        
        return exactly, atleast, atmost

class acqPredict1:
    """
    Class for a estimated probability of the failure for an individual 
    star acquisition 
    """
    def __init__(self, mag, warmpix, beta_results, agasc=""):
        """
        inputs: mag:          Star Magnitude of Acquisition Star
                warmpix:      Estimated Warm Pixel Fraction for Observation
                beta_results: Beta Results loaded from loadacqstats function
                agasc:        agasc ID of star.  defaults to none, needed for
                              plots

        returns: acqPredict1 object

        Available Attributes:
        - agasc:   star agasc ID
        - mag:     star Magnitude
        - warmpix: estimated warm pixel fraction for observation
        - pEsts:   numpy array of the calculated probabilites using the beta
                   draws. probabiltity distribuion.
        - pMean:   mean of all probabilites in pEsts
        - nsims:   total count of probabilities in pEsts
        - pLower:  2.5th percentile of pEsts
        - pUpper:  97.5th percentile of pEsts
        """
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
        """
        Provides a summary of the estimated probability of failure. 
        Includes the estimated mean probability of failure, and a 95 
        percent interval.

        Calls histplot for a summary plot
        """
        
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
        """
        Summary plot of the distribution of probability of failure for a
        given star's magnitude and an estimate of the warm pixel fraction
        """

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
        """
        Builds a numpy array to calculate probabilities.

        inputs: m - star magnitude
                wp - estimated warm pixel fraction
                m_center - mean of the star magnitudes from training data
                wp_center - mean of the warm pixel fraction from the 
                            training data
        """
        mag = m - m_center
        warm = wp - wp_center
        return np.array([1, mag, mag**2, warm, mag*warm, mag*mag*warm])

#############################################################################
# 
# Chandra Star Acquisition Analysis functions
#
#############################################################################

def add_column(recarray, name, val, index=None):
    """
    - Stolen from Ska.Numpy

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
    """
    subsets acquisition dataset by jyear range
    """
    return dset[(dset['tstart_jyear']>=start) & (dset['tstart_jyear']<=end)]

def subset_range_warmpix(dset, wpf):
    """
    subsets acquisition dataset by warm pixel fraction
    """
    return dset[(dset['warm_pix']>=(wpf - 0.01)) & (dset['warm_pix']<=(wpf + 0.01))]