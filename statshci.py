### 
# Author: Shaun Wallace (github - swallace21)
####
# Pulled initial ideas from:
# https://towardsdatascience.com/hypothesis-testing-in-machine-learning-using-python-a0dc89e169ce
###

import pandas as pd
import scipy.stats as sp
import numpy as np

from numpy import mean
from numpy import std
from numpy import var

from numpy.random import seed
from numpy.random import randn

from scipy.stats import shapiro
from scipy.stats import normaltest
from scipy.stats import levene
from scipy.stats import mannwhitneyu
from scipy.stats import wilcoxon
from scipy.stats import kruskal

from math import sqrt

import statsmodels.api as sm

alpha = 0.05

def mean_sd(name,data):
   print('%s (M = %.2f SD = %.2f, N = %.2f, MAX = %.2f, MIN = %.2f)' % (name, mean(data), std(data), len(data), max(data), min(data)))

### standardized effect size
# function to calculate Cohen's d for independent samples
def cohensd(d1,d2):
   # calculate the size of samples
   n1, n2 = len(d1), len(d2)
   # calculate the variance of the samples
   s1, s2 = var(d1, ddof=1), var(d2, ddof=1)
   # calculate the pooled standard deviation
   s = sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
   # calculate the means of the samples
   u1, u2 = mean(d1), mean(d2)
   # calculate the effect size
   d = (u1 - u2) / s
   #print(f"""cohen's d = {d}\n""")
   return d

# for use in chi-square,mcnemar, etc...
def create_contigency_table(d1,d2):
    if isinstance(d1,(list,pd.core.series.Series,np.ndarray)):
        a = np.where(d1 == 1)[0].size 
        b = np.where(d1 == 0)[0].size
        c = np.where(d2 == 1)[0].size
        d = np.where(d2 == 0)[0].size
    else:
        a = d1.count(1)
        b = d1.count(0)
        c = d2.count(1)
        d = d2.count(0)
        
    table = [[a,b],[c,d]]
    # if either b or c is small (b + c < 25) then 
    # chi^2 is not well-approximated by the chi-squared distribution.
    dist = True
    if (b + c) < 25:
        dist = False
    return table,dist


# normality test for normality
def shapiro_wilks(d1,d2):
   """
   data should contain 2 arrays
   """
   # seed the random number generator
   seed(1)
   # generate univariate observations
   data = 5 * randn(100) + 50
   # normality test
   stat, p = shapiro(data)
   #print('shapiro_wilk statistics = %.3f, p = %.3f' % (stat, p))
   
   if p > alpha:
      print('Shapiro-Wilks - Sample looks Gaussian (fail to reject H0)')
      if len(d1) != len(d2):
         levene_equal_variance_test(d1,d2)
      else:
         ttest_repeated_samples(d1,d2)
   else:
      print('Sample does not look Gaussian (reject H0) --- Data is not normally distributed')
      print('The test you should run is: Mann-Whitney U')
      mann_whitney_u(d1,d2)
      exit()


# normality test for D’Agostino’s K^2 test
def dagostino(data):
   """
   data should contain 1 array
   """
   # normality test
   stat, p = normaltest(data)
   print('Statistics=%.3f, p=%.3f' % (stat, p))

   if p > alpha:
      print('Sample looks Gaussian (fail to reject H0)')
   else:
      print('Sample does not look Gaussian (reject H0)')

# variance test
def levene_equal_variance_test(d1,d2):
   """
   data should contain 1-N numpy arrays
   """
   w,p = levene(d1,d2)
   equal_var = False
   if p > alpha:
      #print('Variances are Equal')
      equal_var = True
   ttest_independent(d1,d2,equal_var)

# ttest for comparison of means
def ttest_1sample(data,mean):
   mean_sd('data: ', data)
   df = len(data) - 1
   t, p = sp.ttest_1samp(data, mean(data))
   print(f'one sample t-test \n t({df}) = {round(t,2)}, p = {round(p,4)}\n')


def mann_whitney_u(d1,d2):
   """
   The two samples are combined and rank ordered together. 
   First, determine if the values from the two samples are either:
      Randomly mixed in the rank order, or 
      If they are clustered at opposite ends when combined. 
   A random rank order would mean that the two samples are not different, 
   while a cluster of one sample values would indicate a difference between them.
   """
   print('\nMann Whitney U\n')
   stat, p = mannwhitneyu(d1, d2)
   print('Statistics = %.3f, p = %.3f' % (stat, p))
   # interpret
   alpha = 0.05
   if p > alpha:
      print('Same distribution (fail to reject H0)')
   else:
      print('Different distribution (reject H0)')


def wilcoxon_signed(d1,d2):
   """
   The Wilcoxon signed ranks test is a nonparametric statistical procedure for 
   comparing two samples that are paired, or related. The parametric equivalent 
   to the Wilcoxon signed ranks test goes by names such as the Student’s t-test, 
   t-test for matched pairs, t-test for paired samples, or t-test for dependent samples.
   """
   print('\nWilcoxon Signed-Rank Test\n')
   stat, p = wilcoxon(d1,d2)
   print('Statistics = %.3f, p = %.3f' % (stat, p))
   # interpret
   alpha = 0.05
   if p > alpha:
      print('Same distribution (fail to reject H0)')
   else:
      print('Different distribution (reject H0)')


def kruskal_wallis(d1,d2,d3):
   """
   When the Kruskal-Wallis H-test leads to significant results, then 
   at least one of the samples is different from the other samples. 

   However, the test does not identify where the difference(s) occur. 
   Moreover, it does not identify how many differences occur. 
   To identify the particular differences between sample pairs, 
   a researcher might use sample contrasts, or post hoc tests, 
   to analyze the specific sample pairs for significant difference(s). 
   The Mann-Whitney U-test is a useful method for performing sample contrasts 
   between individual sample sets.
   """
   print('\nKruskal-Wallis H\n')
   stat, p = kruskal(d1,d2,d3)
   print('Statistics = %.3f, p = %.3f' % (stat, p))
   # interpret
   alpha = 0.05
   if p > alpha:
      print('Same distributions (fail to reject H0)')
   else:
      print('Different distributions (reject H0)')


def z_test(d1,d2):
   pass


def fisher_exact(d1,d2):
   pass


def welchs_df(d1,d2):
   df = (d1.var()/d1.size + d2.var()/d2.size)**2 / ((d1.var()/d1.size)**2 / (d1.size-1) + (d2.var()/d2.size)**2 / (d2.size-1))
   # print(f"Welch-Satterthwaite Degrees of Freedom= {df:.4f}")
   return round(df,2)


def ttest_independent(d1,d2,equal_var):
   if equal_var:
      print('\nt-test of equal variance\n')
      df = len(d1) + len(d2) - 2
   else:
      print('\nwelch\'s t-test of unequal variance\n')
      df = welchs_df(d1,d2)

   mean_sd('d1: ',d1)
   mean_sd('d2: ',d2)
   t, p = sp.ttest_ind(d1,d2,equal_var=equal_var)
   d = cohensd(d1,d2)
   print(f't({df}) = {round(t,2)}, p = {round(p,4)}, d = {round(d,3)}\n')


def ttest_repeated_samples(d1,d2):
   print('\nt-test paired samples\n')
   mean_sd('d1: ',d1)
   mean_sd('d2: ',d2)
   df = len(d1) - 1
   t, p = sp.ttest_rel(d1,d2)
   d = cohensd(d1,d2)
   print(f't({df}) = {round(t,2)}, p = {round(p,4)}, d = {round(d,4)}\n')


# McNemar Test - non-parametric binary data
def mcnemar(d1,d2):
   print('\nMcNemar test')
   # Example of calculating the mcnemar test
   from statsmodels.stats.contingency_tables import mcnemar
   table,exact = create_contigency_table(d1,d2)
   result = mcnemar(table, exact=False, correction=True)
   odds_ratio = (table[0][1] / table[0][0]) / (table[1][1] / table[1][0])
   # summarize the finding
   print('$X^2 = %.3f$, $p < %.3f$, $odds ratio = %.3f$' % (result.statistic, result.pvalue, odds_ratio))

if __name__ == "__main__":
   # 1, 2, or more arrays?
   pass
