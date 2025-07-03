import bayesRegress as bR
import numpy as np
import matplotlib.pyplot as plt
import cython
# import makedesign as md


#############################################################################
#
# Finney 1947 Dataset
#
#############################################################################

# finney47 = np.genfromtxt('data/finney1947.csv', dtype=None, delimiter=',', names=True)


# FinneyFunction = "Y ~ Volume + Rate"

# finney47_fit = bR.probitRegression(FinneyFunction, finney47, n_iter=5000, burnin=500, plots=False, identifier="finney47")
# finney47_fit.plotBetas()




############################################################################
#
# Pima Indian Data Set
#
#############################################################################

pimaIndians = np.genfromtxt('data/PimaIndians.csv', dtype=None, delimiter=',', names=True)

pimafunction = "type ~ npreg + glu + bp + skin + bmi + ped + age"
pima_fit = bR.probitRegression(pimafunction, pimaIndians, n_iter=2000, burnin=200, calcIRWLS = False, plots=False, identifier="pima")

#############################################################################
#
# South Africa Dataset
#
#############################################################################

# safr = np.genfromtxt('data/southAfrica.csv', dtype=None, delimiter=',', names=True)

# SAfunction = "chd ~ sbp + tobacco + ldl + famhist + obesity + alcohol + age"

# sa_fit = bR.probitRegression(SAfunction, safr, n_iter=5000, burnin=500, plots=True, identifier="southAfrica")


