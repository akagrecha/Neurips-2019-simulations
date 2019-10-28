import numpy as np
import scipy.integrate as integrate
import sys

def find_mean_lomax(cvar, a, confi):
    """
    Parameters:
    1. cvar: float
        intended CVaR
    2. a: float
        shape parameter
    3. confi: float  
        set confidence between (0,1)
    Returns:
    1. mean: float
        mean of lomax
    """
    omc = 1.0 - confi
    temp = a/((a-1.0)*omc**(1.0/a)) - 1
    mean = cvar/(temp*(a-1))
    return mean

def find_mean_exp(cvar, confi):
    """
    Parameters:
    1. cvar: float
        intended CVaR
    2. confi: float  
        set confidence between (0,1)
    Returns:
    1. mean of Exponential 
    """
    omc = 1-confi
    return cvar/(1.0 - np.log(omc))

def find_truncated_samples(samples, trunc_para):
    """
    Parameters:
    1. samples: np.array(., dtype=float)
        samples of the distribution
    2. trunc_para: float
        truncation parameter
    Returns:
    1. trunc_samples: np.array(., dtype=float)
        max(min(samples, trunc_para), -trunc_para)
    """
    ls = len(samples)
    samples = samples.reshape(-1,1)
    temp1 = np.append(samples, trunc_para*np.ones((ls,1)), axis=1)
    temp1 = np.min(temp1, axis=1).reshape(-1,1)
    temp2 = np.append(temp1, -trunc_para*np.ones((ls,1)), axis=1)
    trunc_samples = np.max(temp2, axis=1)
    return trunc_samples

def find_emp_cvar(samples, confi):
    """
    Parameters: 
    1. samples: np.array(., dtype=float)
        samples of the distribution
    2. confi: float  
        set confidence between (0,1)
    Returns:
    1. cvar_estimate: np.array(., dtype=float)
        Empirical CVaR
    """
    ls = samples.shape[0]
    samples_sorted = np.sort(samples)[::-1]
    ind_ceil = int(np.ceil(ls*(1-confi)))
    ind_floor = int(np.floor(ls*(1-confi)))
    cvar_estimate = (samples_sorted[ind_ceil-1]*(1 - ind_floor/(ls*(1-confi))) 
        + np.sum(samples_sorted[:ind_floor])/(ls*(1-confi)))
    return cvar_estimate

def find_truncated_cvar(samples, trunc_para, confi):
    """
    Parameters: 
    1. samples: np.array(., dtype=float)
        samples of the distribution
    2. trunc_para: float
        truncation parameter
    3. confi: float  
        set confidence between (0,1)
    Returns:
    1. cvar_estimate: float
        truncated empirical cvar (TEC) 
    """
    trunc_samples = find_truncated_samples(samples, trunc_para)
    return find_emp_cvar(trunc_samples, confi)

def gsr(T, iteration):
    """
    ## A basic implementation of Successive Rejects with oblivious TEC ##
    Parameters:
    1. T: int
        Time Horizon
    2. iteration: int
        Keeping track of iterations of gsr
    Returns:
    1. : int
        indicator that mistake happened
    """

    norm = 0.5 + np.sum([1/i for i in range(2,K+1)])
    # n_arr[k] - n_arr[k-1] is the length of kth phase    
    n_arr = np.array([np.floor((T-K)/((K+1-k)*norm)) for k in range(1,K)], dtype=int) 

    # if(iteration%10000 == 0):
   	#     print(iteration)
    optimal_arm = K-1
    # Arms that haven't been rejected
    remaining_set = np.arange(K)
    # Table of samples 
    sample_arms = np.zeros((K, n_arr[-1]))

    # K-1 phases of successive rejects
    for k in np.arange(0,K-1):
        if k == 0:
            n = n_arr[0]  
            for i in remaining_set:
                sample_arms[i,:n] = loss_function(i, n)
        else:
            n = n_arr[k] - n_arr[k-1]
            if n>0:
                for i in remaining_set:
                    sample_arms[i,n_arr[k-1]:n_arr[k]] = loss_function(i, n)
        
        trunc_para = n_arr[k]**q
        estimators = np.array([find_truncated_cvar(sample_arms[i,:n_arr[k]], 
            trunc_para,confi) for i in remaining_set])

        removed_ind = np.argmax(estimators)
        # Arm with greatest estimator value is removed
        removed_arm = remaining_set[removed_ind]
        if removed_arm == optimal_arm:
            # Misidentification because optimal arm removed
            return 1
        remaining_set = np.append(remaining_set[:removed_ind], remaining_set[removed_ind+1:])   
    # the remaining arm will be optimal
    return 0

##################################################
###              Exp(1/mean) Arms              ###
###   Uncomment for the light-tailed setting   ###

confi = 0.95
gap_cvar = 0.15
max_cvar = 3.0
K = 10 # number of arms
max_mean = find_mean_exp(max_cvar, confi)
opt_mean = find_mean_exp(max_cvar-gap_cvar, confi)
def loss_function(arm, num_samples):
    if arm==K-1:
        samples = np.random.exponential(opt_mean, num_samples)
    else:
        samples = np.random.exponential(max_mean, num_samples)
    return samples

###       Light-tailed setting ends Here       ###
##################################################

##################################################
###               Lomax(a,m) Arms              ###
###   Uncomment for the heavy-tailed setting   ###

# a = 2
# confi = 0.95
# gap_cvar = 0.45
# max_cvar = 3.0
# K = 10 # number of arms
# max_mean = find_mean_lomax(max_cvar, a, confi)
# opt_mean = find_mean_lomax(max_cvar-gap_cvar, a, confi)

# def loss_function(arm, num_samples):
#     if arm<K-1:
#         samples = np.random.pareto(a, num_samples)*max_mean*(a-1)
#     else:
#         samples = np.random.pareto(a, num_samples)*opt_mean*(a-1)
#     return samples

###       Heavy-tailed setting Ends Here       ###
##################################################

##########################################
### Lomax(a,mean) and Exp(1/mean) Arms ###
###   Uncomment for the hard setting   ###

# a = 2.0
# confi = 0.95
# gap_cvar = 0.45
# max_cvar = 3.0
# K = 10 # number of arms
# opt_mean = find_mean_exp(max_cvar-gap_cvar, confi)
# lomax_mean = find_mean_lomax(max_cvar, a, confi)
# exp_mean = find_mean_exp(max_cvar, confi)
# def loss_function(arm, num_samples):
#     if arm==K-1:
#         samples = np.random.exponential(opt_mean, num_samples)
#     elif arm<K//2:
#         samples = np.random.pareto(a, num_samples)*lomax_mean*(a-1)
#     else:
#         samples = np.random.exponential(exp_mean, num_samples)
#     return samples

###       Hard setting ends here       ### 
##########################################

# Truncation growth rate
q = float(sys.argv[1])
# Array of time horizons on which SR is evaluated
T_arr = 5000*np.arange(1,11)
# Number of iterations for which SR is run
iters = 50000
# Array of mistakes
mistakes = np.array([[gsr(T, i) 
    for i in range(iters)] for T in T_arr])

# Empirical probability of error
pes = np.mean(mistakes, axis=1)
for pe in pes:
    print(pe)