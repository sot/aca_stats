import copy, sys
import numpy as np 
cimport numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import truncated_norm_C as tnC
import patsy as patc

class probitRegression:
    def __init__(self, function, dset, b=False, v=False, simulate=True, n_iter=5000, burnin=500, calcIRWLS=True, plots=False, identifier='dataset'):
        self.y, self.X, self.betanames = makeDesign(function, dset)
        self.function = function
        self.burnin = burnin
        self.identifier = identifier
        self.npar = len(self.betanames)
        self.simulate = simulate
        self.calcIRWLS = calcIRWLS
        
        #Setting non-informative priors if non were specified
        if not b:
            self.prior_b = np.zeros(self.npar)
        else:
            self.prior_b = b
        
        if not v:
            self.prior_v = 100 * np.identity(self.npar)
        else:
            self.prior_v = v
        
        print """
Output for: {0}
Model     : {1}
            """.format(identifier,function)
        #If simulation set to true, performing bayesian probit regression
        if simulate:
            self.betadraws, self.beta_means = hhProbit(self.y, self.X, self.prior_b, self.prior_v, n_iter)
            self.bayes_betas = np.mean(self.betadraws[burnin:,:], axis=0)
            self.bayes_sd = np.std(self.betadraws[burnin:,:], axis=0)
            self.bayes_lklhd = binLklihood(self.X, self.y, self.bayes_betas, 'probit')
            self.bayes_predict = self.bayes_lklhd.pi
            print """\nSimulation Results ({0} Iterations w/ {1} Burn In):
\nMethod: Bayesian Binary Probit Regression - Holmes and Held (2006)""".format(n_iter, burnin)
            self.printCoefficients(self.bayes_betas, self.bayes_sd, self.betanames)
            

        #Performing Maximum Likelihood Estimates via IRLS
        if calcIRWLS:
            self.ML_betas, self.ML_var = IRWLS(self.y, self.X, link='probit', tol=1e-6, max_iter=100, verbose=True)
            self.ML_sd = np.sqrt(np.diagonal(self.ML_var))
            print """\nMethod: Maximum Likelihood - Iteratively Reweighted Least Squares""".format(n_iter, burnin)
            self.printCoefficients(self.ML_betas, self.ML_sd, self.betanames)
        
        
            self.ML_lklhd = binLklihood(self.X, self.y, self.ML_betas, 'probit')
            self.irwls_predict = self.ML_lklhd.pi

        # self.calcPredictionAccuracy()

        #If they want plots... they'll get plots!
        if plots:
            self.plotBetas()

        

    def calcPredictionAccuracy(self):
        self.bayes_pos_correct = 0.
        self.bayes_neg_correct = 0.
        self.bayes_pos_wrong = 0.
        self.bayes_neg_wrong = 0.
        
        self.irwls_pos_correct = 0.
        self.irwls_neg_correct = 0.
        self.irwls_pos_wrong = 0.
        self.irwls_neg_wrong = 0.
        
        for i in np.arange(len(self.y)):
            if self.simulate:
                if self.y[i] == 1 and self.bayes_predict[i] <= 0.5:
                    self.bayes_pos_wrong += 1.
                if self.y[i] == 1 and self.bayes_predict[i] > 0.5:
                    self.bayes_pos_correct += 1.
                if self.y[i] == 0 and self.bayes_predict[i] >= 0.5:
                    self.bayes_neg_wrong += 1.
                if self.y[i] == 0 and self.bayes_predict[i] < 0.5:
                    self.bayes_neg_correct += 1.


            
            if self.calcIRWLS:
                if self.y[i] == 1 and self.irwls_predict[i] <= 0.5:
                    self.irwls_pos_wrong += 1.
                if self.y[i] == 1 and self.irwls_predict[i] > 0.5:
                    self.irwls_pos_correct += 1.
                if self.y[i] == 0 and self.irwls_predict[i] >= 0.5:
                    self.irwls_neg_wrong += 1.
                if self.y[i] == 0 and self.irwls_predict[i] < 0.5:
                    self.irwls_neg_correct += 1.


        if self.simulate:
            print"""
Bayesian Stats:
Pos. Correct: {0:<6}  Neg. Correct: {1:<6}
Pos. Wrong  : {2:<6}  Neg. Wrong  : {3:<6}
                """.format(self.bayes_pos_correct, self.bayes_neg_correct, self.bayes_pos_wrong, self.bayes_neg_wrong) 
        
        if self.calcIRWLS:    
            print"""
IRWLS Stats:
Pos. Correct: {0:<6}  Neg. Correct: {1:<6}
Pos. Wrong  : {2:<6}  Neg. Wrong  : {3:<6}
                """.format(self.irwls_pos_correct, self.irwls_neg_correct, self.irwls_pos_wrong, self.irwls_neg_wrong) 

        print """
Bayes Classification Rate: {0}
IRWLS Classification Rate: {1}
        """.format((self.bayes_pos_correct + self.bayes_neg_correct)/len(self.y), (self.irwls_pos_correct + self.irwls_neg_correct)/len(self.y))

    def plotBetas(self):
        for beta in np.arange(self.npar):
            betasplot(self.betadraws, self.beta_means, beta, self.burnin, self.identifier, 'beta{0}'.format(beta))

    def printCoefficients(self, est, sds, vnames):
        print "\nCoefficient            Estimate     StdErr     z-Score      pValue"
        for i, n in enumerate(vnames):
            zscore = np.float(est[i]/sds[i])
            pval = np.float(2.*(1. - stats.norm.cdf(np.abs(zscore))))
            print "{0:<20}  {1: 6.3e}  {2: 6.3e}  {3: 6.2e}   {4: 6.3e}".format(n, est[i], sds[i], zscore, pval)
        print
        



def hhProbit(yi, X, b, v, n_iter=10000):
    """
    This code attempts to implement the pseudo-code from the paper "Bayesian Auxiliary Variable Models for Binary and Multinomial Regression" by Chris C. Holmes and Leonhard Held from Bayesian Analysis, 2006.

    Model:

    yi ~ {0,1}. 1 if zi > 0, 0 o/w
    zi ~ xi * beta + epsilon_i
    epsilon_i ~ N(0,1)
    beta ~ Pi(Beta) = N(0,v)

    Inputs: 

    yi: Vector of responses {0,1}
    X: Design Matrix (Must include a vector of 1's if it is necessary for an intercept in the model)
    b: Prior mean on Beta
    v: Prior Variance on Beta
    n_iter: Number of simulations to run

    """
    # Getting the number of observations & parameters
    n_para = X.shape[1]
    n_obs = len(yi)

    #First record constants unaltered within MCMC loop
    V = np.linalg.pinv(np.dot(X.transpose(), X) + np.linalg.pinv(v))
    L = np.linalg.cholesky(V)  
    S = np.dot(V, X.transpose())

    #For j=1 to number of observations
    H = np.empty(dtype=float, shape=(n_obs))
    
    for i in np.arange(n_obs):
        H[i] = np.dot(X[i,:], S[:,i])
    
    W = H / (1 - H)
    Q = W + 1

    # Initialise latent variable Z, from truncated normal
    Z = np.empty(n_obs).transpose()

    for i, y in enumerate(yi):
        if y:
            Z[i] = tnC.rand_tn_a_inf(0,1,0)
        else:
            Z[i] = tnC.rand_tn_inf_b(0,1,0)

    ### Holmes and Held says to initialize Z ~ N(0, I_n)Ind(Y,Z).
    ### Instead of sampling from a multivariate truncated normal,
    ### the above is used since each Zi, Zj is independent by the 
    ### specification of the identity matrix as the variance.
    ### I really hope this assumption holds......

    B = np.dot(S,Z)
    # B denotes the conditional mean of \beta

    betas = np.empty(n_para)
    beta_means = np.empty(n_para)
    
    progress(0., n_iter)    
    progresscheck = int(n_iter * 0.1)

    for i in np.arange(n_iter):
        if (i+1) % progresscheck == 0.:
            progress(i+1, n_iter)
        z_old = copy.copy(Z)
        for j in np.arange(n_obs):
            m = np.dot(X[j,:],B)
            m = m - np.dot(W[j],(Z[j] - m))
            if yi[j]:
                Z[j] = tnC.rand_tn_a_inf(m, Q[j], 0)
            else:
                Z[j] = tnC.rand_tn_inf_b(m, Q[j], 0)
            
            B = B + np.dot((Z[j] - z_old[j]), S[:,j])

        T = stats.multivariate_normal.rvs(np.zeros(n_para), np.identity(n_para), 1).transpose()
        beta_i = (B + np.dot(L,T)).transpose()
        betas = np.vstack((betas, beta_i))

        if i >= 5:
            beta_means = np.vstack((beta_means, np.mean(betas[5:,:], axis=0)))

    betas = betas[2:,:]
    return betas, beta_means

def IRWLS(yi, X, link='logit', tol=1e-8, max_iter=100, verbose=False):
    """
    Iteratively Re-Weighted Least Squares
    """
    nobs, npar = X.shape   
    W = np.identity(nobs) 

    #Ordinary Least Squares as first Beta Guess
    beta_start = betas = wls(X,W,yi)
    
    lstart = lold = binLklihood(X, yi, beta_start, link)
    delt = betaDelta(lold)
    step = 0.0
    while np.sum(np.abs(delt)) > tol and step < max_iter:
        step += 1
        delt = betaDelta(lold)
        lnew = binLklihood(X, yi, betas + delt, link)
        if lnew.likelihood < lold.likelihood:
            delt = delt/2.
            betas = betas + delt
            lold = binLklihood(X,yi,betas,link)
        else:
            betas = betas + delt
            lold = binLklihood(X,yi,betas,link)
        if verbose:
            print """Step {0}: \nLikelihood: {1}""".format(step, lold.likelihood)
    variance = np.linalg.pinv(lold.information) / 4.0
    return betas.transpose()[0], variance

class binLklihood:
    def __init__(self, X, y, betas, link='logit'):
        self.X = X
        self.y = y
        self.betas = betas
        self.link = link
        if link == 'logit':
            self.pi = invlogit(np.dot(X, betas))
        elif link == 'probit':
            self.pi = stats.norm.cdf(np.dot(X, betas))

        self.W = (self.pi * (1 - self.pi))*np.identity(X.shape[0])
        self.likelihood = loglike(self.y, self.pi)
        self.score = X.transpose().dot((self.y - self.pi))
        self.information = X.transpose().dot(self.W).dot(X)
        # self.variance = np.linalg.pinv(self.information) / 4.0


def makeDesign(formula, dataset, intercept=True):
    response, design = patc.dmatrices(formula, dataset, 1)
    if intercept:
        betanames = design.design_info.term_names
        return response, design, betanames
    else:
        betanames =  design.design_info.term_names[1:]
        design = design[:,1:]
        return response, design, betanames

def betaDelta(binlk):
    """
    Change in Delta for a given binomial likelihood object
    """
    return np.linalg.pinv(binlk.information).dot(binlk.score)


def invlogit(val):
    """
    Inverse Logit Function
    """
    return np.exp(val) / (np.exp(val) + 1)

def wls(X, W, yi):
    """
    Weighted Least Squares
    """
    XtWX = X.transpose().dot(W).dot(X) 
    XtWy = X.transpose().dot(W).dot(yi)
    return np.linalg.pinv(XtWX).dot(XtWy)

def loglike(yi, pi):
    """
    Binary Log-Likelihood
    """
    vect_loglike = yi*np.log(pi) + (1-yi)*np.log(1-pi)
    return np.sum(vect_loglike)

def progress(int n, int n_iters):
    cdef float out = np.float(n)/np.float(n_iters) * 100.
    sys.stdout.write("\r{0:.1f}% of iterations complete".format(out))
    sys.stdout.flush()
    return

def betasplot(betas, means, col, burn, dsetname, fname):
    fig = plt.figure()
    plt.subplot(311)
    plt.hist(betas[burn:,col], bins=20, normed=True)
    plt.xlabel('Beta {} Distribution'.format(col))
    plt.subplot(312)
    plt.plot(betas[:,col], marker='', linestyle='-')
    plt.ylabel('Beta {0} Value'.format(col))
    plt.xlabel('Simulation')
    plt.subplot(313)
    plt.plot(means[:,col], marker='', linestyle='-')
    plt.xlabel('Simulation')
    plt.ylabel('Beta{0} Posterior Mean'.format(col))
    fig.set_size_inches(10,10)
    fig.savefig('{0}_{1}.png'.format(dsetname, fname), type='png')
    plt.close()



#############################################################################
#
# Usage Examples
#
#############################################################################


