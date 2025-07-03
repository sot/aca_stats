import numpy as np 
cimport numpy as np
import scipy.stats as stats
import cython
import random as random

cdef extern from "truncated_normal.h":
    double truncated_normal_ab_sample( double mu, double s, double a, double b, int seed)
    double truncated_normal_a_sample ( double mu, double s, double a, int seed )
    double truncated_normal_b_sample ( double mu, double s, double b, int seed )

# cdef double x 
# seed = random.randint(0,2147483647)
# x = truncated_normal_ab_sample(0.,1.,-1.,1, seed)

def rand_tn_ab(mu,s,a,b):
    seed = random.randint(0,2147483647)
    return truncated_normal_ab_sample(mu,s,a,b,seed)

def rand_tn_a_inf(mu,s,a):
    seed = random.randint(0,2147483647)
    return truncated_normal_a_sample(mu,s,a,seed)

def rand_tn_inf_b(mu,s,b):
    seed = random.randint(0,2147483647)
    return truncated_normal_b_sample(mu,s,b,seed)




