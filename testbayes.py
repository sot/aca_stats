import bayesRegress as bR
import numpy as np
import matplotlib.pyplot as plt

finney47 = np.genfromtxt('data/finney1947.csv', dtype=None, delimiter=',', names=True)

yi = finney47['Y']

design = np.vstack((np.ones(len(finney47)), finney47['Volume'], finney47['Rate']))
b = np.zeros(3)
v = 100 * np.identity(3)

b, v = bR.IRWLS(yi,design)

print v

betas, beta_means = bR.hhProbit(yi, design, b, v, 5000)

# pimaIndians = np.genfromtxt('data/PimaIndians.csv', dtype=None, delimiter=',', names=True)

# Pyi = pimaIndians['type']

# Pdesign = np.vstack((np.ones(len(pimaIndians)), pimaIndians['npreg'], pimaIndians['glu'],  pimaIndians['bp'], pimaIndians['skin'], pimaIndians['bmi'], pimaIndians['ped'], pimaIndians['age']))

# # npar = Pdesign.shape[0]

# # Pb = np.zeros(npar)
# # Pv = 100 * np.identity(npar)

# bR.IRWLS(Pyi,Pdesign)

# #finneyBetas = bR.hhProbit(yi, design, b, v, 5000)
# PimaIndianBetas, PimaMeans = bR.hhProbit(Pyi, Pdesign, Pb, Pv, 1000)

safr = np.genfromtxt('data/southAfrica.csv', dtype=None, delimiter=',', names=True)

nobs = len(safr)

yi = safr['chd']

famhist = [1. if i =='Present' else 0. for i in safr['famhist']]

design = np.vstack((np.ones(nobs), 
    safr['sbp'], 
    safr['tobacco'], 
    safr['ldl'],
    famhist,
    safr['obesity'],
    safr['alcohol'], 
    safr['age']))

npara = design.shape[0]

b = np.zeros(npara)
v = 10*np.identity(npara)

# betas, means = bR.hhProbit(yi, design, b, v, 1000)

bR.IRWLS(yi, design)

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
    fig.savefig('betaplots/{0}_{1}.png'.format(dsetname, fname), type='png')
    plt.close()


for i in np.arange(betas.shape[1]):
    betasplot(betas, beta_means, i, 500, "newtest", 'beta{0}'.format(i))

# for i in np.arange(PimaIndianBetas.shape[1]):
#     betasplot(PimaIndianBetas, PimaMeans, i, 500, "test2", 'beta{0}'.format(i))

# for i in np.arange(PimaIndianBetas.shape[1]):
#     betasplot(PimaIndianBetas, i, 500, "pima", 'beta{0}'.format(i))