# Neurips-2019-simulations
Code accompanying the paper "Distribution Oblivious, Risk-Aware Algorithms for Multi-Armed Bandits with Unbounded Rewards"

# Comparison of oblivious and non-oblivious pure exploration bandit algorithms  

In the mean minimization setting, we numerically compare the performance of
successive rejects algorithm using empirical average, oblivious truncated
empirical average and non-oblivious truncated average in three settings, 
viz., when all the arms are light tailed, when all the arms are heavy tailed
and a setting involving both light tailed and heavy tailed arms with the 
optimal arm being light tailed. 

In the CVaR minimization setting, we numerically compare the performance of
successive rejects algorithm using empirical CVaR, oblivious truncated 
empirical CVaR and non-oblivious truncated CVaR in same three settings 
described above.

### Prerequisites

The code was run on a system with the following:

Python 3.6.4
Numpy 1.14.2
Scipy 1.0.0

### Usage

The directory has 6 files, corresponding to the three algorithms for mean
and CVaR minimization each. Each file has three blocks of code corresponding
to the light tailed, heavy tailed and the mixed cases. Uncomment the block
corresponding to the setting needed to be tested and comment the other two 
blocks. Parameters used in the experiments of the paper are present in the 
files. Run the files as follows: <br>
- python cvar_emp.py
- python cvar_non_obl.py
- python cvar_obl.py q
- python mean_emp.py
- python mean_non_obl.py
- python mean_obl.py q <br>
where *q* is the growth parameter of the oblivious truncation parameter. 

### Output

Probability of error of the algorithm evaluated on time horizons specified
in the code.
