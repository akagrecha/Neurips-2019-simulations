import numpy as np
import sys

def find_truncated_samples(samples, trunc_para):
    """
    Parameters:
    1. samples: np.array(., dtype=float)
        samples of the distribution
    2. trunc_para: float
        truncation parameter
    Returns:
    1. trunc_samples: np.array(., dtype=float)
        sample*Indicator(|sample|<=trunc_para)
    """
    trunc_samples = np.copy(samples)
    abs_samples = np.abs(trunc_samples)
    trunc_samples[np.where(abs_samples>trunc_para)] = 0
    return trunc_samples

def find_truncated_mean(samples, trunc_para):
    """
    Parameters:
    1. samples: np.array(., dtype=float)
        samples of the distribution
    2. trunc_para: float
        truncation parameter
    Returns:
    1. : float
        truncated empirical mean (TEM)
    """
    trunc_samples = find_truncated_samples(samples, trunc_para)
    return np.mean(trunc_samples)

def gsr(T,iteration):
    """
    ## A basic implementation of Successive Rejects with oblivious TEA ##
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
        estimators = np.array([find_truncated_mean(sample_arms[i,:n_arr[k]],
            trunc_para) for i in remaining_set])
            
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

gap = 0.03
K = 10
max_mean = 1.0
def loss_function(arm, num_samples):
    if arm==K-1:
        samples = np.random.exponential(max_mean-gap, num_samples)
    else:
        samples = np.random.exponential(max_mean, num_samples)
    return samples

###       Light-tailed setting ends Here       ###
##################################################

##################################################
###               Lomax(a,m) Arms              ###
###   Uncomment for the heavy-tailed setting   ###

# a = 1.8
# gap = 0.1
# K = 10 # Number of arms
# max_mean = 1.0
# def loss_function(arm, num_samples):
#     if arm<K-1:
#         samples = np.random.pareto(a, num_samples)*max_mean*(a-1)
#     else:
#         samples = np.random.pareto(a, num_samples)*(max_mean-gap)*(a-1)
#     return samples

###       Heavy-tailed setting ends Here       ###
##################################################

##########################################
### Lomax(a,mean) and Exp(1/mean) Arms ###
###   Uncomment for the hard setting   ###

# a = 1.8
# gap = 0.1
# K = 10
# max_mean = 1.0
# def loss_function(arm, num_samples):
#     if arm==K-1:
#         samples = np.random.exponential(max_mean-gap, num_samples)
#     elif arm<K//2:
#         samples = np.random.pareto(a, num_samples)*max_mean*(a-1)
#     else:
#         samples = np.random.exponential(max_mean, num_samples)
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