# reliabilityAnalysis
This Reliability Analysis simulates a sample of 30,000 vehicles of 10 vehicle types (Sedan, Minivan, etc) and 3 model years (2019 - 2021).  
Failures data within the 3 year New Vehicle Limited Warranty is simulated assuming Poisson distribution of the # of failures within the NVLW, 
using different failure rates for each vehicle type and model year.  The timing of the failure is assumed to be uniformly distributed across
the observation window, for simplicity.  I assume that the failure rates pick up dramatically in Model Years 2020 and 2021 for both 
Truck and Heavy Duty Truck types.  This way, the differences in rates between vehicle types and model years can be visualized with 
differences in their Mean Cumulative Functions.

**1 create data.py** simulates the vehicle and failures datasets, including the purchase date and assumed NVLW end date.
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

**2 Failure Rate Analysis.py** explores distribution fitting and non-parametric measures of failure rates.  The reliability
package in Python offers a multitude of useful functions for fitting and plotting different curves for the failure rates.

  1) **Survival Functions for First Failures**: The time until first-failure can be assessed using survival methods commonly
     used with medical datasets.  The Kaplan-Meier surival function is a non-parametric curve showing the probability
     that a vehicle will survive (or the particular component will not fail) up to time T.  The method accounts for 
     censoring (in this case, a vehicle's NVLW ending) in order to get an accurate rate of survival probability.
     Confidence intervals are constructed using Greenwood's normal approximation, https://www.math.wustl.edu/~sawyer/handouts/greenwood.pdf
     
     **Example Results from KaplanMeier (95% CI):** taken from https://reliability.readthedocs.io/en/latest/Kaplan-Meier.html#example-1
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
    
    As you can see in the above table, with each censoring row the Survival Probability and KM estimate stay the same as the last row.
    Let $S = # Survivors at Start$ and $P_T = Survival Probability at Fail Time T$.  In this example, there is only 1 failure per fail row.
    Each row's Survival Probability $P_T = \frac{S_T - F_T}{S_T}$, e.g. (25-1)/25=0.96 for time 2754.
    The KM Estimate multiplies the prior row's KME by the new Survival Probability.  E.g. at time 16890, KME = 0.9257 * 0.9565 = 0.885466.
    
  2) 
