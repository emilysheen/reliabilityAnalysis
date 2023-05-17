#########################################################
###              Reliability Analysis                 ###
###                   Emily Sheen                     ###
###  Survival Functions, Distribution Fitting, and    ###
###         Failure Rate Exploratory Analysis         ###
#########################################################

from reliability.Distributions import Weibull_Distribution
from reliability.Fitters import Fit_Weibull_2P
from reliability.Probability_plotting import plot_points
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from reliability.Nonparametric import KaplanMeier
import matplotlib.pyplot as plt

df = pd.read_csv('failures_censors_data.csv').sort_values(by=['vin', 'days_to_censor_fail'])
# pd.set_option('display.max_columns', None)
first_fails = df[df.duplicated(['vin', 'censor_fail_status'])==False]
len(first_fails.loc[first_fails['censor_fail_status'] == 'C'])
check = pd.DataFrame({'number_events' : first_fails.groupby( ['vin', 'censor_fail_status'] ).size()}).reset_index()
max(check['number_events'])

fails = first_fails.loc[first_fails['censor_fail_status'] == 'F', 'days_to_censor_fail'].to_numpy()
censors = first_fails.loc[first_fails['censor_fail_status'] == 'C', 'days_to_censor_fail'].to_numpy()

# FIRST WE WILL EXAMINE THE SURVIVAL FUNCTIONS, ONLY CONSIDERING THE FIRST FAILURES
# This gives us an idea of the percentage of units experiencing failures by Day since purchase_date
# To illustrate the importance of accounting for censoring, we show the same plots ignoring and accounting for censoring.

plt.figure(figsize=(12, 5))
plt.suptitle("Survival Functions for First Failures\nIgnoring Censoring", fontsize=16)
plt.subplot(121)
fit = Fit_Weibull_2P(failures=fails, show_probability_plot=True, print_results=False)  # fits a Weibull distribution to the data and generates the probability plot
plt.xlabel('Days to Failure')
plt.subplot(122)
fit.distribution.SF(label='Weibull SF: Failures Only')  # uses the distribution object from Fit_Weibull_2P and plots the survival function
KaplanMeier(failures=fails, label='Kaplan Meier SF: Failures Only')
plot_points(failures=fails, func='SF', color='yellow')  # overlays the original data on the survival function
plt.xlabel('Days to Failure')
plt.legend()
plt.show()
plt.savefig("plots/KM and Weibull Survival Functions - No Censoring.jpg")

plt.figure(figsize=(12, 5))
plt.suptitle("Survival Functions for First Failures\nWith Censoring", fontsize=16)
plt.subplot(121)
fit2 = Fit_Weibull_2P(failures=fails, right_censored=censors, show_probability_plot=True, print_results=False)  # fits a Weibull distribution to the data and generates the probability plot
plt.xlabel('Days to Failure')
plt.subplot(122)
fit2.distribution.SF(label='Weibull SF: Failures + Right Censoring', xmin=0, xmax=1200)  # uses the distribution object from Fit_Weibull_2P and plots the survival function
KaplanMeier(failures=fails, right_censored=censors, label='Kaplan Meier SF: Failures + Right Censoring')
plot_points(failures=fails, right_censored=censors, func='SF', color='yellow')  # overlays the original data on the survival function
plt.xlabel('Days to Failure')
plt.legend()
plt.show()
plt.savefig("plots/KM and Weibull Survival Functions - With Censoring.jpg")


'''
IGNORING RIGHT CENSORING, THE WEIBULL FIT IS AS FOLLOWS:
Results from Fit_Weibull_2P (95% CI):
Analysis method: Maximum Likelihood Estimation (MLE)
Optimizer: L-BFGS-B
Failures / Right censored: 16597/0 (0% right censored) 
Parameter  Point Estimate  Standard Error  Lower CI  Upper CI
    Alpha         328.463         1.93378   324.694   332.275
     Beta         1.38214      0.00864734   1.36529   1.39919 
Goodness of fit   Value
 Log-likelihood -110194
           AICc  220392
            BIC  220407
             AD 42.8776 
      
TAKING CENSORING INTO CONSIDERATION, THE WEIBULL FIT IS LESS STRONG:       
Results from Fit_Weibull_2P (95% CI):
Analysis method: Maximum Likelihood Estimation (MLE)
Optimizer: TNC
Failures / Right censored: 7540/30000 (79.91476% right censored) 
Parameter  Point Estimate  Standard Error  Lower CI  Upper CI
    Alpha         7776.74         245.773   7309.65   8273.68
     Beta        0.645126      0.00708556  0.631387  0.659164 
Goodness of fit    Value
 Log-likelihood -67893.8
           AICc   135792
            BIC   135809
             AD   117231 
'''
#########################################################################################
### In order to work with other reliability models and incorporate the censoring      ###
###   of our data, we need to convert it to a workable format with reliability.       ###
###  The XCN format takes the following form:                                         ###
###      event_time       censor_code      number_events                              ###
###           13               F                 2                                    ###
###           45               F                 3                                    ###
###           78               F                 1                                    ###
###           90               C                 4                                    ###
###          103               C                 2                                    ###
###   * This format simplifies to the case where number_events == 1 for each row.     ###
###                                                                                   ###
###  * Reliability package allows us to fit several distributions to our sample       ###
###    ~ Functions to fit Non-Location-Shifted Distributions: Fit_Exponential_1P,     ###
###      Fit_Weibull_2P, Fit_Gamma_2P, Fit_Lognormal_2P, Fit_Loglogistic_2P           ###
###      Fit_Normal_2P, Fit_Gumbel_2P, Fit_Beta_2P                                    ###
###    ~ Functions to fit Location-Shifted distributions:  Fit_Exponential_2P,        ###
###      Fit_Weibull_3P, Fit_Gamma_3P, Fit_Lognormal_3P, Fit_Loglogistic_3P           ###
#########################################################################################
df.head

df_xc = df[['days_to_censor_fail','censor_fail_status']].copy().rename(
    columns={'days_to_censor_fail':'event_time', 'censor_fail_status':'censor_code'}).sort_values(
    by=['event_time'], ascending=True)
df_xc.groupby(['event_time', 'censor_code']).size()
df_xcn = pd.DataFrame({'number_events' : df_xc.groupby( ['event_time', 'censor_code'] ).size()}).reset_index()
df_xc.head
df_xcn.head

from reliability.Distributions import Weibull_Distribution
from reliability.Fitters import Fit_Weibull_3P
from reliability.Other_functions import make_right_censored_data, histogram
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity

# Let's try again to fit Weibull distribution incorporating the censoring data using df_xc
fails = (df_xc.loc[df_xc['censor_code'] == 'F', 'event_time']).to_numpy()
censors = (df_xc.loc[df_xc['censor_code'] == 'C', 'event_time']).to_numpy()
wbf = Fit_Weibull_3P(failures=fails,
                     right_censored=censors,
                     show_probability_plot=True, print_results=True)  # fit the Weibull_3P distribution
print('Fit_Weibull_3P parameters:\nAlpha:', wbf.alpha, '\nBeta:', wbf.beta, '\nGamma', wbf.gamma)
# histogram(fails) # generates the histogram of failures, which is uniform until censoring reduces the claims

# Histogram with kernel density estimate
dist_kernel = KernelDensity(bandwidth=40, kernel='gaussian')  # bandwidth = 2 is very choppy
sample = fails.reshape((len(fails), 1))
dist_kernel.fit(sample)
# sample probabilities for a range of outcomes
values = np.asarray([value for value in range(1, 1096)])
values = values.reshape((len(values), 1))
probabilities = dist_kernel.score_samples(values)
probabilities = np.exp(probabilities)
# plot the histogram and pdf
plt.hist(sample, bins=50, density=True)
plt.plot(values[:], probabilities, label='Kernel Density Estimate')
# wbf.distribution.PDF(xmin=0, xmax=1096, label='Fit_Weibull_3P', linestyle='--')  # plots to PDF of the fitted Weibull_3P, not comparable
plt.title('NVLW Failure Emergence:\nFlat until cars leave NVLW')
plt.legend()
plt.show()
plt.savefig("plots/NVLW Failure Emergence.jpg")


'''
* Probability plot is bendy suggesting poor fit.  Makes sense because we did not generate the failures data
using a Weibull generator

Results from Fit_Weibull_3P (95% CI):
Analysis method: Maximum Likelihood Estimation (MLE)
Optimizer: TNC
Failures / Right censored: 16195/30000 (64.94209% right censored) 
Parameter  Point Estimate  Standard Error  Lower CI  Upper CI
    Alpha         2121.91         26.5131   2070.58   2174.52
     Beta        0.827758      0.00601503  0.816052  0.839631
    Gamma          0.9999     4.71258e-05  0.999808  0.999992 
Goodness of fit   Value
 Log-likelihood -136823
           AICc  273651
            BIC  273678
             AD  155736 
'''





### MCF Example
from reliability.Repairable_systems import MCF_nonparametric
from reliability.Datasets import MCF_2
import matplotlib.pyplot as plt
times = [[5, 10, 15, 17], [6, 13, 17, 19], [12, 20, 25, 26], [13, 15, 24], [16, 22, 25, 28]]
MCF_nonparametric(data=times)
plt.show()


from reliability.Repairable_systems import MCF_parametric
times = MCF_2().times
MCF_parametric(data=times)
plt.show()


