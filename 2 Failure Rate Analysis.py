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

df = pd.read_csv('data/failures_censors_data.csv').sort_values(by=['purchase_date', 'vin', 'days_purchase_to_censor_fail']).reset_index(drop=True)
cars = pd.read_csv('data/cars_data.csv').sort_values(by=['purchase_date', 'vin']).reset_index(drop=True)
# pd.set_option('display.max_columns', None)
first_fails = df[df.duplicated(['vin', 'censor_fail_status'])==False]
len(first_fails.loc[first_fails['censor_fail_status'] == 'C'])
check = pd.DataFrame({'number_events' : first_fails.groupby( ['vin', 'censor_fail_status'] ).size()}).reset_index()
max(check['number_events'])

fails = first_fails.loc[first_fails['censor_fail_status'] == 'F', 'days_purchase_to_censor_fail'].to_numpy()/365
censors = first_fails.loc[first_fails['censor_fail_status'] == 'C', 'days_purchase_to_censor_fail'].to_numpy()/365

#####  LET'S CHECK WHICH DISTRIBUTION HAS THE BEST FIT FOR THIS SIMULATED DATA
from reliability.Fitters import Fit_Everything
from reliability.Distributions import Weibull_Distribution
from reliability.Other_functions import make_right_censored_data

# raw_data = Weibull_Distribution(alpha=12, beta=3).random_samples(100, seed=2)  # create some data
# data = make_right_censored_data(raw_data, threshold=14)  # right censor the data
# results = Fit_Everything(failures=data.failures, right_censored=data.right_censored)  # fit all the models
results = Fit_Everything(failures=fails, right_censored=censors) # fit all the models
plt.close()

plt.tight_layout()
results.histogram_plot.savefig("plots/Histogram of All Distribution Fits.jpg", dpi=200)
results.probability_plot.savefig("plots/Probability Plots of All Distribution Fits.jpg")
results.best_distribution_probability_plot.savefig("plots/Probability Plot of Best Dist - " + results.best_distribution_name + ".jpg")
results.PP_plot.savefig("plots/SemiParametric PP Plots of All Distribution Fits.jpg")
print('The best fitting distribution was', results.best_distribution_name)

results.best_distribution.SF(xmin=0, xmax=5, label='Weibull Mixture SF')
plt.xlabel("Years since Purchase")

'''
Results from Fit_Everything:
Analysis method: MLE
Failures / Right censored: 7614/30000 (79.75754% right censored) 
   Distribution   Alpha     Beta    Gamma Alpha 1   Beta 1 Alpha 2   Beta 2 Proportion 1       DS      Mu   Sigma      Lambda  Log-likelihood   AICc    BIC     AD optimizer
Weibull_Mixture                           213.306 0.989545  9891.7 0.987435     0.140727                                             -68153.7 136317 136360 117929       TNC
     Weibull_DS 432.198 0.887733                                                         0.253359                                    -68183.4 136373 136398 117929       TNC
   Lognormal_3P                  0.780559                                                         9.03753 2.89256                    -68199.2 136404 136430 117929       TNC
   Lognormal_2P                                                                                   8.98572 2.83474                    -68219.5 136443 136460 117929       TNC
 Loglogistic_3P 5952.41 0.663814   0.9999                                                                                            -68243.6 136493 136519 117930       TNC
     Weibull_3P 8328.92 0.622596   0.9999                                                                                            -68297.4 136601 136626 117931       TNC
       Gamma_3P 13002.9 0.597674   0.9999                                                                                            -68337.5 136681 136707 117931       TNC
 Loglogistic_2P 5537.27 0.686637                                                                                                       -68399 136802 136819 117930       TNC
     Weibull_2P 7705.37 0.643513                                                                                                     -68468.6 136941 136958 117931       TNC
     Weibull_CR                           7705.35 0.643514 10694.6  15.7975                                                          -68468.6 136945 136979 117931       TNC
       Gamma_2P 11631.5  0.62014                                                                                                     -68517.3 137039 137056 117932       TNC
 Exponential_2P                    0.9999                                                                         0.000298313        -69419.6 138843 138860 117960       TNC
 Exponential_1P                                                                                                   0.000355937        -69559.1 139120 139129 117949  L-BFGS-B
      Normal_2P                                                                                   1536.84  895.94                    -73822.6 147649 147666 117956       TNC
      Gumbel_2P                                                                                   1655.77 562.319                    -74666.2 149336 149353 117958       TNC 


The best fitting distribution was Weibull_Mixture
'''




# EXAMINE THE SURVIVAL FUNCTIONS, ONLY CONSIDERING THE FIRST FAILURES
# This gives us an idea of the percentage of units experiencing failures by Day since purchase_date
# To illustrate the importance of accounting for censoring, we show the same plots ignoring and accounting for censoring.

plt.figure(figsize=(12, 5))
plt.suptitle("Survival Functions for First Failures\nIgnoring Censoring", fontsize=16)
plt.subplot(121)
fit = Fit_Weibull_2P(failures=fails, show_probability_plot=True, print_results=False)  # fits a Weibull distribution to the data and generates the probability plot
# plt.xlabel('Days to Failure')
plt.subplot(122)
fit.distribution.SF(label='Weibull SF: Failures Only')  # uses the distribution object from Fit_Weibull_2P and plots the survival function
KaplanMeier(failures=fails, label='Kaplan Meier SF: Failures Only')
plot_points(failures=fails, func='SF', color='yellow')  # overlays the original data on the survival function
plt.xlabel('Years to Failure')
plt.legend()
plt.show()
plt.savefig("plots/KM and Weibull Survival Functions - No Censoring.jpg")
plt.close()

plt.figure(figsize=(12, 5))
plt.suptitle("Survival Functions for First Failures\nWith Censoring", fontsize=16)
plt.subplot(121)
fit2 = Fit_Weibull_2P(failures=fails, right_censored=censors, show_probability_plot=True, print_results=False)  # fits a Weibull distribution to the data and generates the probability plot
# plt.xlabel('Days to Failure')
plt.subplot(122)
fit2.distribution.SF(label='Weibull SF: Failures + Right Censoring', xmin=0, xmax=3)  # uses the distribution object from Fit_Weibull_2P and plots the survival function
KaplanMeier(failures=fails, right_censored=censors, label='Kaplan Meier SF: Failures + Right Censoring')
plot_points(failures=fails, right_censored=censors, func='SF', color='yellow')  # overlays the original data on the survival function
plt.xlabel('Years to Failure')
plt.legend()
plt.show()
plt.savefig("plots/KM and Weibull Survival Functions - With Censoring.jpg")
plt.close()

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
###  * This format simplifies to the case where number_events == 1 for each row.      ###
###                                                                                   ###
###  * MULTIPLE EVENTS PER VEHICLE: Since we have multiple events per vehicle in      ###
###    some cases, we will assume the repair/replacement fixed the issue on the       ###
###    same day that the failure happened, for simplicity.  Generally repairs could   ###
###    take days or weeks and we would want to start the clock again once the car     ###
###    is back in service.  After each repair, we reset the clock so that the next    ###
###    event (or censoring) is based on days since last repair/replacement.           ###
###                                                                                   ###
###  * Reliability package allows us to fit several distributions to our sample       ###
###    ~ Functions to fit Non-Location-Shifted Distributions: Fit_Exponential_1P,     ###
###      Fit_Weibull_2P, Fit_Gamma_2P, Fit_Lognormal_2P, Fit_Loglogistic_2P           ###
###      Fit_Normal_2P, Fit_Gumbel_2P, Fit_Beta_2P                                    ###
###    ~ Functions to fit Location-Shifted distributions:  Fit_Exponential_2P,        ###
###      Fit_Weibull_3P, Fit_Gamma_3P, Fit_Lognormal_3P, Fit_Loglogistic_3P           ###
#########################################################################################
df.head

# Using purchase date rather than resetting clock after repair/replacement
df_xc_buy = df[['days_purchase_to_censor_fail','censor_fail_status']].copy().rename(
    columns={'days_purchase_to_censor_fail':'event_time', 'censor_fail_status':'censor_code'}).sort_values(
    by=['event_time'], ascending=True).reset_index(drop=True)
df_xc_buy.groupby(['event_time', 'censor_code']).size()
df_xcn_buy = pd.DataFrame({'number_events' : df_xc_buy.groupby( ['event_time', 'censor_code'] ).size()}).reset_index()
df_xc_buy.head
df_xcn_buy.head

from reliability.Distributions import Weibull_Distribution
from reliability.Fitters import Fit_Weibull_3P
from reliability.Other_functions import make_right_censored_data, histogram
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity

# Let's try again to fit Weibull distribution incorporating the censoring data using df_xc
fails = (df_xc_buy.loc[df_xc_buy['censor_code'] == 'F', 'event_time']).to_numpy()
censors = (df_xc_buy.loc[df_xc_buy['censor_code'] == 'C', 'event_time']).to_numpy()
wbf = Fit_Weibull_3P(failures=fails,
                     right_censored=censors,
                     show_probability_plot=True, print_results=True)  # fit the Weibull_3P distribution
print('Fit_Weibull_3P parameters:\nAlpha:', wbf.alpha, '\nBeta:', wbf.beta, '\nGamma', wbf.gamma)
# histogram(fails) # generates the histogram of failures, which is uniform until censoring reduces the claims
plt.close()

# Histogram with kernel density estimate
dist_kernel = KernelDensity(bandwidth=30, kernel='exponential')  # bandwidth = 2 is very choppy
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
plt.xlabel("Days Since Vehicle Purchase")
# wbf.distribution.PDF(xmin=0, xmax=1096, label='Fit_Weibull_3P', linestyle='--')  # plots to PDF of the fitted Weibull_3P, not comparable
plt.title('NVLW Failure Emergence for Days since Vehicle Purchase')
plt.legend()
plt.show()
plt.savefig("plots/NVLW Failure Emergence for Days since Vehicle Purchase.jpg")
plt.close()

# Let's try again to fit Weibull distribution now treating repeat claims as independent events (restarting the clock after repair/replacement)
df_xc = df[['days_since_event','censor_fail_status']].copy().rename(
    columns={'days_since_event':'event_time', 'censor_fail_status':'censor_code'}).sort_values(
    by=['event_time'], ascending=True).reset_index(drop=True)
df_xc.groupby(['event_time', 'censor_code']).size()
df_xcn = pd.DataFrame({'number_events' : df_xc.groupby( ['event_time', 'censor_code'] ).size()}).reset_index()
df_xc.head
df_xcn.head
fails = (df_xc.loc[df_xc['censor_code'] == 'F', 'event_time']).to_numpy()
censors = (df_xc.loc[df_xc['censor_code'] == 'C', 'event_time']).to_numpy()
wbf = Fit_Weibull_3P(failures=fails,
                     right_censored=censors,
                     show_probability_plot=True, print_results=True)  # fit the Weibull_3P distribution
print('Fit_Weibull_3P parameters:\nAlpha:', wbf.alpha, '\nBeta:', wbf.beta, '\nGamma', wbf.gamma)
# histogram(fails) # generates the histogram of failures, which is uniform until censoring reduces the claims
plt.close()

# Histogram with kernel density estimate
dist_kernel = KernelDensity(bandwidth=30, kernel='exponential')  # bandwidth = 2 is very choppy
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
plt.xlabel("Days Since Last Event")
# wbf.distribution.PDF(xmin=0, xmax=1096, label='Fit_Weibull_3P', linestyle='--')  # plots to PDF of the fitted Weibull_3P, not comparable
plt.title('NVLW Failure Emergence for Days since Last Event')
plt.legend()
plt.show()
plt.savefig("plots/NVLW Failure Emergence for Days since Last Event.jpg")
plt.close()

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

#########################################################################################
###    *  Mean Cumulative Function for Average Cumulative # Failures on Vehicles      ###
###                                                                                   ###
###    In order to study multiple failures per vehicle within the NVLW, the           ###
###      Mean Cumulative Function (MCF) is a nonparametric measurement similar to     ###
###     the Kaplan Meier survival curve, but it shows average # cum failures over     ###
###     the vehicles' lifetimes, rather than time until failure (first failure)       ###
###   DATA FORMAT needs to be a list of lists, with each inner list representing      ###
###      a single vehicle, the last time being the censor time for that car           ###
#########################################################################################

### MCF Example
from reliability.Repairable_systems import MCF_nonparametric, MCF_parametric
import matplotlib.pyplot as plt

times = []
for vin in df['vin'].unique():
    vin_times = (df.loc[df['vin'] == vin, 'days_purchase_to_censor_fail']).tolist()
    times.append(vin_times)

MCF_nonparametric(data=times)
plt.title("Non-Parametric MCF for All Vehicles")
plt.xlabel("Days since NVLW Start")
plt.show()
plt.savefig("plots/NonParametric MCF for All Vehicles.jpg")
plt.close()

MCF_parametric(data=times)
plt.xlabel("Days since NVLW Start")
plt.show()
plt.savefig("plots/Parametric MCF for All Vehicles.jpg")

# Parametric plot here looks like a poor fit to the data



#  Now we want to try to figure out WHY we have such high repair rates early in the vehicles' lives
#  We have a few other synthesized variables we can look at by left joining in our car_data.csv file

fails = pd.read_csv('data/failures_censors_data.csv').sort_values(by=['purchase_date', 'vin', 'days_purchase_to_censor_fail']).reset_index(drop=True)
cars = pd.read_csv('data/cars_data.csv').sort_values(by=['purchase_date', 'vin']).reset_index(drop=True)

dat = cars.merge(fails, on=['vin', 'purchase_date', 'n_fails', 'nvlw_end_date'], how='left').sort_values(
    by=['purchase_date', 'vin', 'days_purchase_to_censor_fail']).reset_index(drop=True)

# pd.set_option('display.max_columns', None)
dat.head()
dat.columns

car_types = dat['car_type'].unique()
model_years = dat['model_year'].unique()

# for cart in car_types:
#     for my in model_years:
#         globals()['time_%s' % cart + str(my)] = 'Hello'

import re
import math
regex = re.compile('[^a-zA-Z]')
#First parameter is the replacement, second parameter is your input string
# regex.sub('', 'ab3d*E')
#Out: 'abdE'

for cart in car_types:
    for my in model_years:
        cart_name = regex.sub('', cart)
        ldat = dat.loc[(dat['model_year'] == my) & (dat['car_type'] == cart), ['vin', 'annual_mileage', 'days_purchase_to_censor_fail']].sort_values(
            by=['vin', 'days_purchase_to_censor_fail']).reset_index(drop=True)
        ltimes = []
        lmiles = []
        for vin in ldat['vin'].unique():
            vin_years = ((ldat.loc[ldat['vin'] == vin, 'days_purchase_to_censor_fail'])/365).tolist()
            vin_kmiles = ((ldat.loc[ldat['vin'] == vin, 'days_purchase_to_censor_fail'])/1095*(ldat.loc[ldat['vin'] == vin, 'annual_mileage'])*3/1000).tolist()
            ltimes.append(vin_years)
            lmiles.append(vin_kmiles)
        globals()['years_%s' % str(my) + cart_name] = ltimes
        globals()['kmiles_%s' % str(my) + cart_name] = lmiles
        # Plot and save both MCF's
        plt.figure(figsize=(12, 5))
        plt.suptitle("MCFs (Years) for " + str(my) + " " + cart, fontsize=16)
        plt.subplot(121)
        MCF_nonparametric(data=ltimes)
        plt.xlabel('Years since NVLW Start')
        plt.title('Non-Parametric MCF')
        plt.subplot(122)
        MCF_parametric(data=ltimes)
        plt.title("Parametric MCF")
        plt.xlabel("Years since NVLW Start")
        plt.show()
        plt.savefig("plots/MCFs by Vehicle and MY/MCF in Years for " + str(my) + " "+ cart + ".jpg")
        plt.close()
        plt.figure(figsize=(12, 5))
        plt.suptitle("MCFs (Miles) for " + str(my) + " " + cart, fontsize=16)
        plt.subplot(121)
        MCF_nonparametric(data=lmiles)
        plt.xlabel('Miles (000\'s) since NVLW Start')
        plt.title('Non-Parametric MCF')
        plt.subplot(122)
        MCF_parametric(data=lmiles)
        plt.title("Parametric MCF")
        plt.xlabel("Miles (000\'s) since NVLW Start")
        plt.show()
        plt.savefig("plots/MCFs by Vehicle and MY/MCF in Miles for " + str(my) + " " + cart + ".jpg")
        plt.close()

# Let's try to make plots with 2019 - 2021 all shown on the same plot for each car_type
# LEFT = Years Sedan MY 19-21, RIGHT = Miles Sedan MY 19-21
# Try first with Sedan

from math import ceil
from matplotlib.lines import Line2D

# First make a dataset of max values
cart_maxes = pd.DataFrame({'car_type': car_types, 'cart_mcf_year_max': None, 'cart_mcf_kmile_max': None,})
for index, row in cart_maxes.iterrows():
    cart = row['car_type']
    cart_name = regex.sub('', cart)
    s19 = MCF_nonparametric(data=globals()['years_2019%s' % cart_name], print_results=False, show_plot=False).results
    s19 = s19.loc[s19['MCF'] != '']
    s20 = MCF_nonparametric(data=globals()['years_2020%s' % cart_name], print_results=False, show_plot=False).results
    s20 = s20.loc[s20['MCF'] != '']
    s21 = MCF_nonparametric(data=globals()['years_2021%s' % cart_name], print_results=False, show_plot=False).results
    s21 = s21.loc[s21['MCF'] != '']
    uplim_year = float(ceil(max(s21['MCF_upper'].max(), s20['MCF_upper'].max(), s19['MCF_upper'].max())*100)/100)
    cart_maxes.at[index, 'cart_mcf_year_max'] = uplim_year
    s19 = MCF_nonparametric(data=globals()['kmiles_2019%s' % cart_name], print_results=False, show_plot=False).results
    s19 = s19.loc[s19['MCF'] != '']
    s20 = MCF_nonparametric(data=globals()['kmiles_2020%s' % cart_name], print_results=False, show_plot=False).results
    s20 = s20.loc[s20['MCF'] != '']
    s21 = MCF_nonparametric(data=globals()['kmiles_2021%s' % cart_name], print_results=False, show_plot=False).results
    s21 = s21.loc[s21['MCF'] != '']
    uplim_kmiles = ceil(max(s21['MCF_upper'].max(), s20['MCF_upper'].max(), s19['MCF_upper'].max()) * 100) / 100
    cart_maxes.at[index, 'cart_mcf_kmile_max'] = uplim_kmiles


for index, row in cart_maxes.iterrows():
    cart = row['car_type']
    cart_name = regex.sub('', cart)
    ymax = row['cart_mcf_year_max']
    mmax = row['cart_mcf_kmile_max']
    plt.figure(figsize=(12, 5))
    plt.suptitle("MCFs for 2019 - 2021 " + cart + "s", fontsize=16)
    plt.subplot(121)
    MCF_nonparametric(data=globals()['years_2019%s' % cart_name], color='red', print_results=False)
    MCF_nonparametric(data=globals()['years_2020%s' % cart_name], color='green', print_results=False)
    MCF_nonparametric(data=globals()['years_2021%s' % cart_name], color ='blue', print_results=False, show_plot=True)
    plt.xlabel('Years since NVLW Start')
    plt.title('Nonparametric MCF (Years)')
    plt.xlim(0, 3.1)
    plt.ylim(0, ymax)
    plt.legend([Line2D([0], [0], color='red', lw=2),
                Line2D([0], [0], color='green', lw=2),
                Line2D([0], [0], color='blue', lw=2)], ['2019', '2020', '2021'])
    plt.show()
    plt.subplot(122)
    # max_miles = math.ceil(max(dat.loc[dat['car_type'] == 'Sedan', 'annual_mileage'])*3/1000)
    MCF_nonparametric(data=globals()['kmiles_2019%s' % cart_name], color='red', print_results=False)
    MCF_nonparametric(data=globals()['kmiles_2020%s' % cart_name], color='green', print_results=False)
    MCF_nonparametric(data=globals()['kmiles_2021%s' % cart_name], color='blue', print_results=False)
    plt.title("Nonparametric MCF (Miles)")
    plt.xlabel("Miles (000\'s) since NVLW Start")
    plt.xlim(0, 36.5)
    plt.ylim(0, mmax)
    plt.legend([Line2D([0], [0], color='red', lw=2),
                Line2D([0], [0], color='green', lw=2),
                Line2D([0], [0], color='blue', lw=2)], ['2019', '2020', '2021'])
    plt.show()
    plt.savefig("plots/MCFs by Vehicle and MY/Nonparametric MCFs for 2019-21 "+cart_name+".jpg")



# After splitting up the MCF's for each model year and vehicle type, we see that Trucks and Heavy Duty Trucks have
# the largest rates of failure during the 3 year limited warranty.
# FROM THESE MCF PLOTS, we see that the problem starts to pick up in 2020 model year trucks, and persists in the 2021 model year

years_2019nontrucks = []
years_2020nontrucks = []
years_2021nontrucks = []
kmiles_2019nontrucks = []
kmiles_2020nontrucks = []
kmiles_2021nontrucks = []
non_truck = ['Minivan', 'Convertible', 'Mid-Size SUV', 'Electric Vehicle', 'Hatchback', 'Compact SUV', 'Large SUV', 'Sedan']
for cart in non_truck:
    cart_name = regex.sub('', cart)
    years_2019nontrucks = years_2019nontrucks + globals()['years_2019%s' % cart_name]
    years_2020nontrucks = years_2020nontrucks + globals()['years_2020%s' % cart_name]
    years_2021nontrucks = years_2021nontrucks + globals()['years_2021%s' % cart_name]
    kmiles_2019nontrucks = kmiles_2019nontrucks + globals()['kmiles_2019%s' % cart_name]
    kmiles_2020nontrucks = kmiles_2020nontrucks + globals()['kmiles_2020%s' % cart_name]
    kmiles_2021nontrucks = kmiles_2021nontrucks + globals()['kmiles_2021%s' % cart_name]
years_2019trucks = years_2019Truck + years_2019HeavyDutyTruck
years_2020trucks = years_2020Truck + years_2020HeavyDutyTruck
years_2021trucks = years_2021Truck + years_2021HeavyDutyTruck
kmiles_2019trucks = kmiles_2019Truck + kmiles_2019HeavyDutyTruck
kmiles_2020trucks = kmiles_2020Truck + kmiles_2020HeavyDutyTruck
kmiles_2021trucks = kmiles_2021Truck + kmiles_2021HeavyDutyTruck


plt.figure(figsize=(12, 5))
plt.suptitle("Nonparametric MCFs for 2019 - 2021 Trucks and Non-Trucks", fontsize=16)
plt.subplot(121)
MCF_nonparametric(data=years_2019nontrucks, color='red', print_results=False)
MCF_nonparametric(data=years_2020nontrucks, color='green', print_results=False)
MCF_nonparametric(data=years_2021nontrucks, color ='blue', print_results=False, show_plot=True)
plt.xlabel('Years since NVLW Start')
plt.title('Non-Trucks MCF (Years)')
plt.xlim(0, 3.1)
plt.legend([Line2D([0], [0], color='red', lw=2),
            Line2D([0], [0], color='green', lw=2),
            Line2D([0], [0], color='blue', lw=2)], ['2019', '2020', '2021'])
plt.show()
plt.subplot(122)
MCF_nonparametric(data=years_2019trucks, color='red', print_results=False)
MCF_nonparametric(data=years_2020trucks, color='green', print_results=False)
MCF_nonparametric(data=years_2021trucks, color='blue', print_results=False)
plt.title("Trucks MCF (Years)")
plt.xlabel("Years since NVLW Start") # Miles (000\'s) since NVLW Start")
plt.xlim(0, 3.1)
# plt.ylim(0, mmax)
plt.legend([Line2D([0], [0], color='red', lw=2),
            Line2D([0], [0], color='green', lw=2),
            Line2D([0], [0], color='blue', lw=2)], ['2019', '2020', '2021'])
plt.show()
plt.savefig("plots/MCFs by Vehicle and MY/Nonparametric MCFs (Years) for 2019-21 Trucks and Non-Trucks.jpg")


plt.figure(figsize=(12, 5))
plt.suptitle("Nonparametric MCFs (Miles) for 2019 - 2021 Trucks and Non-Trucks", fontsize=16)
plt.subplot(121)
MCF_nonparametric(data=kmiles_2019nontrucks, color='red', print_results=False)
MCF_nonparametric(data=kmiles_2020nontrucks, color='green', print_results=False)
MCF_nonparametric(data=kmiles_2021nontrucks, color ='blue', print_results=False, show_plot=True)
plt.xlabel('Miles (000\'s) since NVLW Start')
plt.title('Non-Trucks MCF (K Miles)')
plt.xlim(0, 36.1)
plt.legend([Line2D([0], [0], color='red', lw=2),
            Line2D([0], [0], color='green', lw=2),
            Line2D([0], [0], color='blue', lw=2)], ['2019', '2020', '2021'])
plt.show()
plt.subplot(122)
MCF_nonparametric(data=kmiles_2019trucks, color='red', print_results=False)
MCF_nonparametric(data=kmiles_2020trucks, color='green', print_results=False)
MCF_nonparametric(data=kmiles_2021trucks, color='blue', print_results=False)
plt.title("Trucks MCF (K Miles)")
plt.xlabel("Miles (000\'s) since NVLW Start") # Miles (000\'s) since NVLW Start")
plt.xlim(0, 36.1)
# plt.ylim(0, mmax)
plt.legend([Line2D([0], [0], color='red', lw=2),
            Line2D([0], [0], color='green', lw=2),
            Line2D([0], [0], color='blue', lw=2)], ['2019', '2020', '2021'])
plt.show()
plt.savefig("plots/MCFs by Vehicle and MY/Nonparametric MCFs (K Miles) for 2019-21 Trucks and Non-Trucks.jpg")

