# reliabilityAnalysis
This Reliability Analysis simulates a sample of 30,000 vehicles of 10 vehicle types (Sedan, Minivan, etc) and 3 model years (2019 - 2021).  
Failures data within the 3 year New Vehicle Limited Warranty is simulated assuming Poisson distribution of the # of failures within the NVLW, 
using different failure rates for each vehicle type and model year.  The timing of the failure is assumed to be uniformly distributed across
the observation window, for simplicity.  I assume that the failure rates pick up dramatically in Model Years 2020 and 2021 for both 
Truck and Heavy Duty Truck types.  This way, the differences in rates between vehicle types and model years can be visualized with 
differences in their Mean Cumulative Functions.

## 1 create data.py 
simulates the vehicle and failures datasets, including the purchase date and assumed NVLW end date.
New Vehicle Limited Warranties (NVLW's) are termed as 36 months / 36,000 miles (whichever comes sooner).  In order to simulate
correct censoring dates, I first randomly generate annual mileage, using unique mean/SD mileage for each car type and model year.  
The NVLW end date is assumed to be the minimum of the time-out date (3 years from purchase date) and the mile-out date (assuming 
uniform annual mileaging patterns).  In practice, one challenge of time-to-failure modeling is the lack of information regarding 
how each vehicle's mileage accrues.  A dealer can see the car's mileage only when it comes in for service/maintenance, so if the 
car goes on a long road trip or experiences a lull in driving during the cold winter, this mileage is unknown to the manufacturer
until the next service appointment.  Mileage more accurately predicts wear and tear on a vehicle than time, because there is 
huge variability in driving patterns for different vehicle owners.  A work truck might accrue over 50K miles per year, 
while a remote worker without a commute could drive under 5K miles annually. In simulating the data, we assume large standard 
deviations in these mileage patterns to capture the variability in driving across vehicle owners.  MCF's based on time (days since
purchase) and mileage (1K miles since purchase) can paint a very different picture for how failures accrue.

## 2 Failure Rate Analysis.py
explores distribution fitting and non-parametric measures of failure rates.  The reliability
package in Python offers a multitude of useful functions for fitting and plotting different curves for the failure rates.

  1) **Kaplan-Meier Survival Functions for First Failures**: The time until first-failure can be assessed using survival methods commonly
     used with medical datasets.  The **Kaplan-Meier survival function** is a non-parametric curve showing the probability
     that a vehicle will survive (or the particular component will not fail) up to time T.  The method accounts for 
     censoring (in this case, a vehicle's NVLW ending) in order to get an accurate rate of survival probability.
     Confidence intervals are constructed using Greenwood's normal approximation, https://www.math.wustl.edu/~sawyer/handouts/greenwood.pdf
     
     **Example Results from KaplanMeier (95% CI):** taken from https://reliability.readthedocs.io/en/latest/Kaplan-Meier.html#example-1
      
      As you can see in the table below, with each censoring row the Survival Probability and KM estimate stay the same as the last row.
      Let $S = $\# Survivors at Start and $P_T =$ Survival Probability at Fail Time $T$.  In this example, there is only 1 failure per 
      fail row. Each row's Survival Probability $P_T = \frac{S_T - F_T}{S_T}$, e.g. $(25-1)/25=0.96$ for $T= 2754$.
      The KM Estimate multiplies the prior row's KME by the new Survival Probability.  E.g. at $T=16890$, $KME = 0.9257 * 0.9565 = 0.885466$.
    
     ```
        Fail    Censor=0/    # Survivors  Surv Prob.        Kaplan-Meier Estimate    LowerCI   UpperCI**
       Times    Failure=1      at Start
        3961         0            31          1                              1          1          1
        4007         0            30          1                              1          1          1
        4734         0            29          1                              1          1          1
        5248         1            28       0.964286                      0.964286   0.895548       1
        6054         0            27       0.964286                      0.964286   0.895548       1
        7298         0            26       0.964286                      0.964286   0.895548       1
        7454         1            25         0.96        0.96 * 0.9643 = 0.925714   0.826513       1
       10190         0            24         0.96                        0.925714   0.826513       1
       16890         1            23       0.956522    0.9257 * 0.9565 = 0.885466    0.76317       1
       17200         1            22       0.954545    0.8855 * 0.9545 = 0.845217   0.705334    0.985101
       23060         0            21       0.954545                      0.845217   0.705334    0.985101
     ```  
  2) **The Weibull Distribution** is a parameterized probability distribution function that assumes failure rate is proportional to a 
    power of time.  The data is used to optimize the $\alpha$ scale parameter and $\beta$ shape parameter to fit the data to the parameterized
    curve.  See https://reliability.readthedocs.io/en/latest/Equations%20of%20supported%20distributions.html for more information.
    To illustrate the importance of accounting for censoring, **2 Failure Rate Analysis.py** shows plots both ignoring and including the 
    censoring information.  Without including censoring, at the end of the NVLW, 100% of vehicles are assumed to have experienced a failure.
    With censoring, over 70% of vehicles survive without a first failure at the end of their NVLWs.
  3) **Related Probability Distributions** The Weibull distribution is related to the Exponential and Gamma distributions.  The broadest
    distribution is the generalized Gamma, https://en.wikipedia.org/wiki/Generalized_gamma_distribution.  If the two shape parameters
    in the generalized Gamma distribution are equal, it simplifies to the Weibull distribution.  If the Weibull's shape parameter $\beta = 1$,
    this indicates that the failure rate is constant over time, and the Weibull simplifies to the exponential distribution. $\beta > 1$ 
    indicates that the failure rate increases over time, while $\beta < 1$ indicates the failure rate decreases over time.  The reliability
    package in Python allows you to fit your data to several distributions at once, including Exponential_1P, Exponential_2P, Gamma_2P, 
    Gamma_3P, Gumbel_2P, Loglogistic_2P, Loglogistic_3P, Lognormal_2P, Lognormal_3P, Normal_2P, Weibull_2P, Weibull_3P, Weibull_CR, Weibull_DS, 
    and Weibull_Mixture, described https://reliability.readthedocs.io/en/latest/Fitting%20a%20specific%20distribution%20to%20data.html.
  4) **The Weibull Mixture Model** When we fit all the simulated failure data with reliability.Fitters.Fit_Everything, the optimal model fit
    turns out to be the Weibull Mixture model.  Mixture models work by combining multiple PDF's or CDF's at proportions summing to 1.  They
    are discussed in more detail here: https://reliability.readthedocs.io/en/latest/Mixture%20models.html.  Mixture models are useful when 
    the failure data seems to come in groups. For example, if multiple types of failures are considered (e.g. Engine and Clutch), or different
    types and model years of vehicles are included, then mixture models may be appropriate.  In this case, since I designed the data
    such that Trucks and Heavy Duty Trucks experienced much higher failure rates in the 2020 and 2021 model years, 
