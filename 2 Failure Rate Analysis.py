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

df = pd.read_csv('failures_censors_data.csv')
pd.set_option('display.max_columns', None)
fails = df.loc[df['censor_fail_status'] == 'F', 'days_to_censor_fail'].to_numpy()
censors = df.loc[df['censor_fail_status'] == 'C', 'days_to_censor_fail'].to_numpy()

# Let's look at the difference in estimates if we ignore the right censoring
# dist = Weibull_Distribution(alpha=30, beta=2)  # creates the distribution object
# data = dist.random_samples(20, seed=42)  # draws 20 samples from the distribution. Seeded for repeatability
plt.subplot(121)
fit = Fit_Weibull_2P(failures=fails, show_probability_plot=False, print_results=False)  # fits a Weibull distribution to the data and generates the probability plot
plt.subplot(122)
fit.distribution.SF(label='fitted distribution')  # uses the distribution object from Fit_Weibull_2P and plots the survival function
# dist.SF(label='original distribution', linestyle='--') # plots the survival function of the original distribution
# Plotting survival functions with and without censoring
KaplanMeier(failures=fails, right_censored=censors, label='Failures + Right Censors')
KaplanMeier(failures=fails, label='Failures Only')
plot_points(failures=fails, func='SF')  # overlays the original data on the survival function
plt.legend()
plt.show()
plt.savefig("output.jpg")

plt.title('Kaplan-Meier estimates showing the\nimportance of including censored data')
plt.xlabel('Days to Failure')
plt.legend()
plt.show()

'''
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
'''
#########################################################################################
### In order to incorporate the censoring of our data, we need to convert it to a     ###
###   workable format with reliability.  The XCN format takes the following form:     ###
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
                     # right_censored=censors,
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
wbf.distribution.PDF(xmin=0, xmax=1096, label='Fit_Weibull_3P', linestyle='--')  # plots to PDF of the fitted Weibull_3P
plt.title('Fitting comparison for failures and right censored data')
plt.legend()
plt.show()


dist_kernel.fit(sample)
# dist.PDF(label='True Distribution')  # plots the true distribution's PDF
plt.show()

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



a = 30
b = 2
g = 20
threshold=55
dist = Weibull_Distribution(alpha=a, beta=b, gamma=g) # generate a weibull distribution
raw_data = dist.random_samples(500, seed=2)  # create some data from the distribution
data = make_right_censored_data(raw_data,threshold=threshold) #right censor some of the data
print('There are', len(data.right_censored), 'right censored items.')
wbf = Fit_Weibull_3P(failures=data.failures, right_censored=data.right_censored, show_probability_plot=False, print_results=False)  # fit the Weibull_3P distribution
print('Fit_Weibull_3P parameters:\nAlpha:', wbf.alpha, '\nBeta:', wbf.beta, '\nGamma', wbf.gamma)
histogram(raw_data,white_above=threshold) # generates the histogram using optimal bin width and shades the censored part as white
dist.PDF(label='True Distribution')  # plots the true distribution's PDF
wbf.distribution.PDF(label='Fit_Weibull_3P', linestyle='--')  # plots to PDF of the fitted Weibull_3P
plt.title('Fitting comparison for failures and right censored data')
plt.legend()
plt.show()



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


