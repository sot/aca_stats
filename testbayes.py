import bayesRegress as bR
import numpy as np

finney47 = np.genfromtxt('data/finney1947.csv', dtype=None, delimiter=',', names=True)

yi = finney47['Y']

design = np.vstack((np.ones(len(finney47)), finney47['Volume'], finney47['Rate']))
b = np.zeros(3)
v = 100 * np.identity(3)


# pimaIndians = np.genfromtxt('data/PimaIndians.csv', dtype=None, delimiter=',', names=True)

# yi = pimaIndians['type']

# design = np.vstack((np.ones(len(pimaIndians)), pimaIndians['npreg'], pimaIndians['glu'],  pimaIndians['bp'], pimaIndians['skin'], pimaIndians['bmi'], pimaIndians['ped'], pimaIndians['age']))

# npar = design.shape[0]
# print npar
# b = np.zeros(npar)
# v = 100 * np.identity(npar)


bR.hhProbit(yi, design, b, v, 5000)