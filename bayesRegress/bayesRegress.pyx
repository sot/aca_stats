import copy, sys
import numpy as np 
cimport numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import truncated_norm_C as tnC

# @profile
def hhProbit(np.ndarray[long, ndim=1] yi, np.ndarray[double, ndim=2] X, np.ndarray[double, ndim=1] b, np.ndarray[double, ndim=2] v, int n_iter=10000):
    X = X.transpose()
    yi = yi.transpose()
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
    cdef int n_para = X.shape[1]
    cdef int n_obs = len(yi)

    #First record constants unaltered within MCMC loop
    cdef np.ndarray[double, ndim=2] V = np.linalg.pinv(np.dot(X.transpose(), X) + np.linalg.pinv(v))
    cdef np.ndarray[double, ndim=2] L = np.linalg.cholesky(V)  
    cdef np.ndarray[double, ndim=2] S = np.dot(V, X.transpose())

    #For j=1 to number of observations
    cdef np.ndarray[double, ndim=1] H = np.empty(dtype=float, shape=(n_obs))
    
    for i in np.arange(n_obs):
        H[i] = np.dot(X[i,:], S[:,i])
    
    cdef np.ndarray[double, ndim=1] W = H / (1 - H)
    cdef np.ndarray[double, ndim=1] Q = W + 1

    # Initialise latent variable Z, from truncated normal
    cdef np.ndarray[double, ndim=1] Z = np.empty(n_obs).transpose()

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

    cdef np.ndarray[double, ndim=1] B = np.dot(S,Z)
    # B denotes the conditional mean of \beta

    # n_iter = 10000
    low, mid, high = float('-inf'), 0., float('inf')
    betas = np.empty(n_para)
    beta_means = np.empty(n_para)
    
    cdef np.ndarray[double, ndim=1] z_old
    cdef float m
    

    for i in np.arange(n_iter):
        if (i+1) % 1000. == 0.:
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
        # print np.mean(betas, axis=0)
        betas = np.vstack((betas, beta_i))
        if i >= 5:
            beta_means = np.vstack((beta_means, np.mean(betas[5:,:], axis=0)))
    print "\n{0} Simulations complete".format(n_iter)
    betas = betas[1:,:]
    print betas[0:10,:]
    print beta_means[-1,:]
    return betas, beta_means

def IRWLS(yi, X, link='logit', tol=1e-2, max_iter=100, verbose=False):
    """
    Iteratively Re-Weighted Least Squares
    """
    npar, nobs = X.shape
    X = X.transpose()
    yi = yi.transpose()     
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
    variance = np.linalg.pinv(lold.information)
    return betas, variance

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
        self.W = np.diag(self.pi*(1 - self.pi))
        self.likelihood = loglike(self.y, self.pi)
        self.score = X.transpose().dot((y - self.pi))
        self.information = X.transpose().dot(self.W).dot(X)


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

# def hhProbit(np.ndarray[long, ndim=1] yi, np.ndarray[double, ndim=2] X, np.ndarray[double, ndim=1] b, np.ndarray[double, ndim=2] v, int n_iter=10000):
#     """
#     This code attempts to implement the pseudo-code from the paper "Bayesian Auxiliary Variable Models for Binary and Multinomial Regression" by Chris C. Holmes and Leonhard Held from Bayesian Analysis, 2006.

#     Model:

#     yi ~ {0,1}. 1 if zi > 0, 0 o/w
#     zi ~ xi * beta + epsilon_i
#     epsilon_i ~ N(0,1)
#     beta ~ Pi(Beta) = N(0,v)

#     Inputs: 

#     yi: Vector of responses {0,1}
#     X: Design Matrix (Must include a vector of 1's if it is necessary for an intercept in the model)
#     b: Prior mean on Beta
#     v: Prior Variance on Beta
#     n_iter: Number of simulations to run

#     """
#     X = X.transpose()
#     yi = yi.transpose()

#     # Getting the number of observations & parameters
#     cdef int n_para = X.shape[1]
#     cdef int n_obs = len(yi)

#     #First record constants unaltered within MCMC loop
#     V = np.linalg.pinv(np.dot(X.transpose(), X) + np.linalg.pinv(v))
#     L = np.linalg.cholesky(V)  
#     S = np.dot(V, X.transpose())
    
#     #For j=1 to number of observations
#     cdef np.ndarray[double, ndim=1] H = np.empty(dtype=float, shape=(n_obs))
    

#     for i in xrange(n_obs):
#         H[i] = np.dot(X[i,:], S[:,i])
#     print H
#     # H = (X * S).diagonal().transpose()
#     cdef np.ndarray[double, ndim=1] W = H / (1 - H)
#     cdef np.ndarray[double, ndim=1] Q = W + 1


#     # Initialise latent variable Z, from truncated normal
#     cdef np.ndarray[double, ndim=1] Z = np.empty(dtype=float, shape=n_obs)

#     for i, y in enumerate(yi):
#         if y:
#             Z[i] = stats.truncnorm.rvs(0., float('inf'), 0, 1)
#         else:
#             Z[i] = stats.truncnorm.rvs(float('-inf'), 0., 0, 1)

    
#     # ### Holmes and Held says to initialize Z ~ N(0, I_n)Ind(Y,Z).
#     # ### Instead of sampling from a multivariate truncated normal,
#     # ### the above is used since each Zi, Zj is independent by the 
#     # ### specification of the identity matrix as the variance.
#     # ### I really hope this assumption holds......

#     cdef np.ndarray[double, ndim=1] B = np.dot(S,Z)
#     # B denotes the conditional mean of \beta

#     # n_iter = 10000
#     cdef double low = float('-inf')
#     cdef double mid = 0.0
#     cdef double high = float('inf')

#     cdef np.ndarray[double, ndim=2] betas = np.empty(dtype='float', shape=(1, n_para))
    
#     cdef np.ndarray[double, ndim=1] z_old = Z

#     for i in xrange(n_iter):
#         if (i+1) % 1000. == 0.:
#             progress(i+1, n_iter)
        
#         z_old = Z

#         for j in xrange(n_obs):
#             m = np.dot(X[j,:],B)
#             m = m - np.dot(W[j],(Z[j] - m))
#             if yi[j]:
#                 Z[j] = stats.truncnorm.rvs((mid - m) / Q[j], (high - m) / Q[j], loc=m, scale=Q[j])
#             else:
#                 Z[j] = stats.truncnorm.rvs((low - m) / Q[j], (mid - m) / Q[j], loc=m, scale=Q[j])

#             B = B + np.dot((Z[j] - z_old[j]), S[:,j])

#         T = stats.multivariate_normal.rvs(np.zeros(n_para), np.identity(n_para), 1).transpose()
#         beta_i = (B + np.dot(L,T)).transpose()
#         # print beta_i
#         betas = np.vstack((betas, beta_i))
#     print "\n{0} Simulations complete".format(n_iter)
#     betas = betas[1:,:]
#     print betas[0:,:]

def progress(int n, int n_iters):
    cdef float out = np.float(n)/np.float(n_iters) * 100.
    sys.stdout.write("\r{0:.1f}% of iterations complete".format(out))
    sys.stdout.flush()
    return

#############################################################################
#
# Examples and Test Data
#
#############################################################################

# # # #### Testing Finney Dataset
# finney47 = np.genfromtxt('data/finney1947.csv', dtype=None, delimiter=',', names=True)

# yi = finney47['Y']

# design = np.vstack((np.ones(len(finney47)), finney47['Volume'], finney47['Rate']))
# b = np.zeros(3)
# v = 100 * np.identity(3)

# #### Testing Pima Indians Dataset
# # pimaIndians = np.genfromtxt('data/PimaIndians.csv', dtype=None, delimiter=',', names=True)

# # yi = np.matrix(pimaIndians['type']).T

# # design = np.matrix(np.vstack((np.ones(len(pimaIndians)), pimaIndians['npreg'], pimaIndians['glu'],  pimaIndians['bp'], pimaIndians['skin'], pimaIndians['bmi'], pimaIndians['ped'], pimaIndians['age'])).transpose())

# # print design.shape[1]

# # b = np.matrix(np.zeros(8)).transpose()
# # v = np.matrix(100 * np.identity(8))

# hhProbit(yi, design, b, v, 10000)

# np.array([i + 1 if i==1 else i-1 for i in test])