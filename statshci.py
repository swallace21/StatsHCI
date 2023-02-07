### 
# Author: Shaun Wallace (github - swallace21)
####
# Pulled initial ideas from:
# https://towardsdatascience.com/hypothesis-testing-in-machine-learning-using-python-a0dc89e169ce
###

import pandas as pd
import scipy.stats as sp
import numpy as np
from itertools import combinations

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
from scipy.stats import chi2_contingency
from statsmodels.sandbox.stats.multicomp import multipletests

from math import sqrt

alpha = 0.05

def mean_sd(name,data):
   print('\t%s (M = %.4f SD = %.2f, N = %.2f, MAX = %.2f, MIN = %.2f)' % (name, mean(data), std(data), len(data), max(data), min(data)))

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

### effect size
def effectSize(d1,d2):
   print('Effect Size d1 / d2 = %.3f' % ((mean(d1)/mean(d2))))

### effect size
def effectSizeNonParametric(d1,d2,U1):
   # see "Examples"
   # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.mannwhitneyu.html
   nx, ny = len(d1), len(d2)
   U2 = nx*ny - U1
   U = min(U1, U2)
   N = nx + ny
   z = (U - nx*ny/2 + 0.5) / np.sqrt(nx*ny * (N + 1)/ 12)
   es = z/sqrt((nx+ny))
   print(f'z-value = {z:4.2f} and the effect size of Mann Whitney U = {es:4.2f}')

def get_difference(current, previous, currName, prevName):
    if current == previous:
         print('no difference')
    try:
         changeType = 'increase'
         if current < previous:
               changeType = 'decrease'
         diff = round((current / previous), 4)
         if diff < 1.5:
            change = (abs(current - previous) / previous) * 100.0
            print(f'\tThere is a {change:4.2f}% {changeType} between {currName} and {prevName}.')
         else:
            print(f'\t{currName} is {diff:4.2f} times greater than {prevName}.')
    except ZeroDivisionError:
         return float('inf')

def difference(current, previous, currName, prevName):
   get_difference(current, previous, currName, prevName)
   get_difference(previous, current, prevName, currName)

# for use in chi-square,mcnemar, etc...
def create_contigency_table(d1,d2,d1Name = 'Group 1',d2Name = 'Group 2'):
   if isinstance(d1,(pd.core.series.Series,np.ndarray)):
      a = np.where(d1 == 1)[0].size 
      b = np.where(d1 == 0)[0].size
      c = np.where(d2 == 1)[0].size
      d = np.where(d2 == 0)[0].size
   else:
      a = d1.count(1)
      b = d1.count(0)
      c = d2.count(1)
      d = d2.count(0)
   if a > 0 and b > 0 and c > 0 and d > 0:
      avg1 = (a/(a+b)) * 1.0
      avg2 = (c/(c+d)) * 1.0
      #print(avg1,avg2)
      #print(f'{d1Name} = {avg1:4.3f} :: {d2Name} = {avg2:4.3f}')
      if avg1 > 0.0 and avg2 > 0.0:
         difference(avg1,avg2,d1Name,d2Name)
      table = [[a,b],[c,d]]
      # if either b or c is small (b + c < 25) then 
      # chi^2 is not well-approximated by the chi-squared distribution.
      dist = True
      if (b + c) < 25:
         dist = False
      return table,dist,True
   else:
      return False

def normality(d1,d2):
   """
   data should contain 1 array
   """
   # normality test
   data = d1 + d2
   if len(data) > 5000:
      stat, p = normaltest(data)
      print('Dagostino: Statistics=%.3f, p=%.3f' % (stat, p))
   else:
      stat, p = shapiro(data)
      print('Shapiro Wilks: statistic = %.3f, p = %.3f' % (stat, p))

   if p > alpha:
      print('Sample looks Gaussian (fail to reject H0)')
      if len(d1) != len(d2):
         levene_equal_variance_test(d1,d2)
      else:
         ttest_repeated_samples(d1,d2)
   else:
      print('Sample does not look Gaussian (reject H0)')
      mann_whitney_u(d1,d2)

# normality test for normality
def shapiro_wilks(d1,d2):
   """
   data should contain 2 arrays
   """
   data = d1 + d2
   # normality test
   stat, p = shapiro(data)
   print('shapiro_wilks: statistic = %.3f, p = %.3f' % (stat, p))
   
   if p > alpha:
      print('Shapiro-Wilks - Sample looks Gaussian (fail to reject H0)')
      if len(d1) != len(d2):
         levene_equal_variance_test(d1,d2)
      else:
         ttest_repeated_samples(d1,d2)
   else:
      print(f'Sample is not Gaussian (reject H0) - Data is not normally distributed - use Mann-Whitney U')
      mann_whitney_u(d1,d2)


# normality test for D’Agostino’s K^2 test
def dagostino(d1,d2):
   """
   data should contain 1 array
   """
   # normality test
   data = d1 + d2
   stat, p = normaltest(data)
   print('Statistics=%.3f, p=%.3f' % (stat, p))

   if p > alpha:
      print('Sample looks Gaussian (fail to reject H0)')
      if len(d1) != len(d2):
         levene_equal_variance_test(d1,d2)
      else:
         ttest_repeated_samples(d1,d2)
   else:
      print('Sample does not look Gaussian (reject H0)')
      mann_whitney_u(d1,d2)

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
   print('Mann Whitney U')
   mean_sd('d1: ',d1)
   mean_sd('d2: ',d2)
   difference(mean(d1), mean(d2), 'd1', 'd2')
   stat, p = mannwhitneyu(d1,d2)
   effectSizeNonParametric(d1,d2,stat)
   print('Statistics = %.3f, p = %.6f' % (stat, p))
   # interpret
   alpha = 0.05
   if p > alpha:
      print('Same distribution (fail to reject H0)')
   else:
      print('Different distribution (reject H0)')
      print(f'A Mann Whitney U test reveals this <effectSize> in <metric> is statistically significant, $U$ = {int(stat)}, $p$ < {p}.')
   # A Mann Whitney U test reveals this 30.5\% increase in accuracy per visit in 2022 is statistically significant, $U$ = 7928, $p$ < 0.001.


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
def mcnemar(d1,d2,d1Name = 'Group 1',d2Name = 'Group 2'):
   print('McNemar test')
   # Example of calculating the mcnemar test
   from statsmodels.stats.contingency_tables import mcnemar
   table,exact,valid = create_contigency_table(d1,d2,d1Name,d2Name)
   if valid:
      result = mcnemar(table, exact=exact, correction=True)
      print(table)
      odds_ratio = (table[0][1] / table[0][0]) / (table[1][1] / table[1][0])
      # summarize the finding
      df = 1
      print('$X^2(%.0i) = %.3f$, $p < %.3f$, $odds ratio = %.3f$' % (df, result.statistic, result.pvalue, odds_ratio))
   else:
      print('Length of the groups is 0 -- cannot create contigency table')

def conversion_rate(d1y,d1n):
   cr = (d1y/(d1y+d1n))
   print('\t\tConversion Rate: = %.3f' % (cr))
   return cr

def chi2(d1y,d1n,d2y,d2n):
   print('\nCHI Squared test')
   print(d1y,d1n,d2y,d2n)
   d1_sum = (d1y+d1n)
   d2_sum = (d2y+d2n)
   print('Conversion Rate D1: = %.3f' % ((d1y/d1_sum)))
   print('Conversion Rate D2: = %.3f' % ((d2y/d2_sum)))

   from scipy.stats import chi2_contingency
   observed = [
      [d1y,d1n],
      [d2y,d2n],
   ]
   chi2, p, dof, expected = chi2_contingency(observed)
   print('\tchi-square:', chi2)
   print('\tp-value:', p)
   print('\tdegree of freedom:', dof)
   #print('\texpected value table:', expected)

   if p >= 0.05:
      print('H0 is accepted')
   else:
      print('H0 is rejected')
      if p < 0.001:
         p = f'p < 0.001'
      else:
         p = f'p = {p:4.3f}'
      print(f'There is a significant relationship between the two variables, <var1> are more likely than <var2> to <action>, $X^2 ({dof}$, $N = {(d1_sum+d2_sum)})$ = ${chi2:4.2f}$, {p}.')

def chisq_multiple_and_posthoc_corrected(df):
    """Receives a dataframe and performs chi2 test and then post hoc.
    Prints the p-values and corrected p-values (after FDR correction)"""
    # start by running chi2 test on the matrix
    chi2, p, dof, ex = chi2_contingency(df, correction=True)
    print(f"Chi2 result of the contingency table: {chi2}, p-value: {p} :: $X^2 ({dof}$, $N = {df.values.sum()})$ = ${chi2:4.2f}$, {p}.")
    
    # post-hoc
    all_combinations = list(combinations(df.index, 2))  # gathering all combinations for post-hoc chi2
    p_vals = []
    chi2_vals = []
    print("Significance results:")
    for comb in all_combinations:
        new_df = df[(df.index == comb[0]) | (df.index == comb[1])]
        chi2, p, dof, ex = chi2_contingency(new_df, correction=True)
        p_vals.append(p)
        chi2_vals.append(chi2)
        # print(f"For {comb}: {p}")  # uncorrected

    # checking significance
    # correction for multiple testing
    reject_list, corrected_p_vals = multipletests(p_vals, method='fdr_bh')[:2]
    for p_val, corr_p_val, reject, comb in zip(p_vals, corrected_p_vals, reject_list, all_combinations):
        print(f"{comb}: p_value: {p_val:5f}; corrected: {corr_p_val:5f} ({p_val}) reject: {reject}")
        if reject:
            print(f"""There is a significant relationship between {comb[0]} and {comb[1]}, {comb[0]} are more likely than {comb[0]} to <action>, $X^2 ({dof}$, $N = <sum__both_vars>)$ = ${chi2:4.2f}$, {p}.\n""")
        else:
            print()

if __name__ == "__main__":
   # 1, 2, or more arrays?
   pass
