#########################################################
###              Reliability Analysis                 ###
###                   Emily Sheen                     ###
###                Weibull Analysis                   ###
#########################################################


from reliability.Distributions import Weibull_Distribution
from reliability.Fitters import Fit_Weibull_2P
from reliability.Probability_plotting import plot_points
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv ('claims_censors_data.csv')
claims = df.loc[df['censor_claim_status'] == 'Claim']

# First we don't assume any right censoring
# dist = Weibull_Distribution(alpha=30, beta=2)  # creates the distribution object
# data = dist.random_samples(20, seed=42)  # draws 20 samples from the distribution. Seeded for repeatability
plt.subplot(121)
fit = Fit_Weibull_2P(failures=claims['days_to_censor_claim'].to_numpy(),print_results=True)  # fits a Weibull distribution to the data and generates the probability plot
plt.subplot(122)
fit.distribution.SF(label='fitted distribution')  # uses the distribution object from Fit_Weibull_2P and plots the survival function
dist.SF(label='original distribution', linestyle='--') # plots the survival function of the original distribution
plot_points(failures=data, func='SF')  # overlays the original data on the survival function
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
df.head

from reliability.Distributions import Weibull_Distribution
from reliability.Fitters import Fit_Weibull_3P
from reliability.Other_functions import make_right_censored_data, histogram
import matplotlib.pyplot as plt

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


