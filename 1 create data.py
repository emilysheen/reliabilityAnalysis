#########################################################
###              Reliability Analysis                 ###
###                   Emily Sheen                     ###
###        1) Create automobile contract data         ###
###        2) Create repair failures data             ###
#########################################################

### Assume we have a sample of 10,000 vehicles from model years 2019, 2020, and 2021 with 36 month/36,000 mile bumper to bumper NVLW's.

import faker
import numpy as np
import pandas as pd
import datetime
from faker import Faker
fake = Faker()
import random
# fake.date_between(start_date='today', end_date = '+30y')
from reliability.Distributions import Weibull_Distribution
from reliability.Fitters import Fit_Weibull_2P
from reliability.Probability_plotting import plot_points
import matplotlib.pyplot as plt

random.seed(7)
vin = [x for x in range(30000)]
manufacture_date = [fake.date_between(datetime.date(2018, 8, 1), datetime.date(2019,8,1)) for x in range(10000)] + \
            [fake.date_between(datetime.date(2019, 8, 1), datetime.date(2020,8,1)) for x in range(10000, 20000)] + \
            [fake.date_between(datetime.date(2020, 8, 1), datetime.date(2021,8,1)) for x in range(20000, 30000)]
def buy_date(dttime):
    add_days = random.randint(60, 360)
    return dttime + datetime.timedelta(days=add_days)

# Calling purchase_date method on each manufacture date to add a random number of days between 60 and 360
purchase_date = [buy_date(i) for i in manufacture_date]

model_year = [2019 for x in range(10000)] + [2020 for x in range(10000, 20000)] + [2021 for x in range(20000, 30000)]
cart = ["Sedan" for x in range(1000)] + ["Minivan" for x in range(1000, 2000)] + \
           ["Truck" for x in range(2000, 3000)] + ["Hatchback" for x in range(3000, 4000)] + \
           ["Heavy Duty Truck" for x in range(4000,5000)] + ["Compact SUV" for x in range(5000,6000)] + \
           ["Mid-Size SUV" for x in range(6000,7000)] + ["Large SUV" for x in range(7000,8000)] + \
           ["Convertible" for x in range(8000,9000)] + ["Electric Vehicle" for x in range(9000,10000)]
car_type = cart + cart + cart

cars = pd.DataFrame({"vin": vin, "manufacture_date": manufacture_date, "purchase_date": purchase_date,
                     "model_year": model_year, "car_type": car_type})

print(cars)

# Now we want to add approximate annual mileage and assume different means by car_type to generate random avg mileage
car_type_mileage = pd.DataFrame({"car_type": cars['car_type'].unique(),
                                 "avg_annual_mileage": [11000, 13000, 20000, 15000, 30000, 15000, 14000, 13000, 9000, 7000],
                                 "sd_annual_mileage":  [2000,  2500,  4000,  3000,  8000, 3000, 2500, 2500, 1500, 1000]})

# We will assume the car type average mileages are normally distributed with above mean and SD
# We will draw each individual car's average annual mileage from ~N(avg_annual_mileage, sd_annual_mileage)
len(cars.loc[cars['car_type'] == 'Sedan'])
def get_miles(carty):
    avg = car_type_mileage.loc[car_type_mileage['car_type'] == carty, 'avg_annual_mileage']
    sd =  car_type_mileage.loc[car_type_mileage['car_type'] == carty, 'sd_annual_mileage']
    car_mileage = abs(np.random.normal(avg, sd, 3000))
    return car_mileage

for cart in car_type_mileage['car_type']:
    print(cart)
    cars.loc[cars['car_type'] == cart, 'annual_mileage'] = get_miles(cart)

# Double check that average mileage makes sense
print(cars.groupby('car_type').agg(
    avg_mileage = pd.NamedAgg(column='annual_mileage', aggfunc=np.mean),
    sd_mileage = pd.NamedAgg(column='annual_mileage', aggfunc=np.std)
))
print(car_type_mileage.sort_values(by='car_type'))
# Looks good

# Let's generate a random number of engine block failures for each vehicle in our sample
# Set lambda values for each car type and model year

lambdas = pd.DataFrame({"car_type": cars['car_type'].unique().tolist() * 3,
                        "model_year": [2019]*10 + [2020]*10 + [2021]*10,
                        "lambda": [0.1, 0.2, 0.3, 0.1, 0.5, 0.1, 0.2, 0.2, 0.1, 0.01,
                                   0.01, 0.3,  3, 0.1,  3,  0.2, 0.3, 0.3, 0.1, 0.01,
                                   0.01, 0.2,  3, 0.2,  3,  0.2, 0.2, 0.2, 0.1, 0.01]})
print(lambdas)

print(cars.groupby(['car_type', 'model_year']).agg(
    avg_mileage = pd.NamedAgg(column='annual_mileage', aggfunc=np.mean),
    sd_mileage = pd.NamedAgg(column='annual_mileage', aggfunc=np.std),
    n_cars = pd.NamedAgg(column='vin', aggfunc='count')
))


def get_fail_counts(carty, my):
    lam = lambdas.loc[(lambdas['car_type'] == carty) & (lambdas['model_year'] == my), 'lambda']
    n_fails = np.random.poisson(lam, size=1000)
    return n_fails

get_fail_counts('Sedan', 2020)

for index, row in lambdas.iterrows():
    cart = row['car_type']
    my = row['model_year']
    cars.loc[(cars['car_type'] == cart) & (cars['model_year'] == my), 'n_fails'] = get_fail_counts(cart, my)

# n_fails will occur within the NVLW period of 3 years / 36,000 miles
# Need to add timeout_date and mileage_out_date and min of those is nvlw_end_date
cars['nvlw_timeout'] = cars['purchase_date'] + pd.DateOffset(months=36)
cars['days_to_mileout'] = (36000/(cars['annual_mileage']*3)*365*3)

# Define a function to add days_to_mileout to purchase_date to get nvlw_mileout
def add_days (row):
    mileout_days = round(row['days_to_mileout'])
    mileout = row['purchase_date'] + datetime.timedelta(days=mileout_days)
    mileout = pd.to_datetime(mileout, format='%Y-%m-%d')
    return(mileout)

cars['nvlw_mileout'] = cars.apply(lambda row: add_days(row), axis=1)
cars['nvlw_end_date'] = cars[['nvlw_mileout','nvlw_timeout']].min(axis=1)

cars.to_csv("cars_data.csv", header=True, index=False)

########################################################################################################################
######  2) REPAIR Failures DATA:  Let's assume we were approached about problematic engine block failures leading  #####
######     to extensive NVLW claims.  We are interested in discovering the root cause of the engine block problems,#####
######     which vehicle types and model years are afflicted with the issue, and whether a recall is necessary.    #####
######  *     Since for this example, we are designing the dataset to be explored, we will assume this is a 2020   #####
######     and 2021 problem for trucks and heavy duty trucks.  We will assume Poisson distribution with lambda = 3 #####
######     for 2020 and 2021 trucks and heavy duty trucks.  Let lambda values vary for other vehicle types and     #####
######     model years, between 0.01 and 0.5.                                                                      #####
######  *     Let lambda = 3 for Engine Block claims w/in the 3 yr / 36K mi NVLW period, both trucks MY 2020/21    #####
######  *     Let lambda < 1 for all other car types, all 3 model years                                            #####
########################################################################################################################


#### Now we need to determine which day in the NVLW period the failures will occur.

fails = cars[["vin", "purchase_date", 'nvlw_end_date', "n_fails"]]
               # "manufacture_date",  "model_year", "car_type",
               # "annual_mileage", 'nvlw_timeout', 'days_to_mileout', 'nvlw_mileout',


fails.loc[(fails['nvlw_end_date'] < pd.to_datetime(datetime.date.today(), format='%Y-%m-%d')), 'censor_date'] = \
    fails.loc[fails['nvlw_end_date'] < pd.to_datetime(datetime.date.today(), format='%Y-%m-%d'), 'nvlw_end_date']
fails.loc[fails['nvlw_end_date'] >= pd.to_datetime(datetime.date.today(), format='%Y-%m-%d'), 'censor_date'] = \
    pd.to_datetime(datetime.date.today(), format='%Y-%m-%d')

fails.head(100)
print(fails.dtypes)
fails.loc[fails['nvlw_end_date'] >= pd.to_datetime(datetime.date.today(), format='%Y-%m-%d'), 'censor_date'].unique()

fails[['censor_date', 'purchase_date']] = fails[['censor_date', 'purchase_date']].apply(pd.to_datetime)
fails['days_purchase_to_censor'] = (fails['censor_date'] - fails['purchase_date']).dt.days

# If the vehicle has 0 claims, we just need the row for censor date
# If a vehicle has 1+ claim, we need a row for each claim and the censor date
# Let's split up the censor / claim data to simplify transformation
fails = fails.loc[fails.index.repeat(fails.n_fails + 1)]  # duplicates rows for n_fails + 1, sorts by vin and purchase_date
fails = fails.sort_values(by=['purchase_date', 'vin'], axis=0)

censors = fails[fails.duplicated()==False] # 1st copy of each vehicle to get censor date
len(censors)
fails = fails[fails.duplicated()]  # duplicate copies for claims == claims[claims['n_fails'] >= 1]

# For the censor data, just set the date to the censor date from earlier and the status to Censor
censors['days_purchase_to_censor_fail'] = censors['days_purchase_to_censor']
censors['censor_fail_status'] = 'C'
censors['censor_fail_date'] = censors['censor_date']

# We need to generate random failure dates, we will assume uniform distribution for simplicity
fails['censor_fail_status'] = 'F'

# Check for small windows in [purchase_date, censor_date] and negative days_purchase_to_censor values
len(fails.loc[fails['days_purchase_to_censor'] < 0]) == 0

pd.set_option('display.max_columns', None)
len(cars.loc[cars['nvlw_end_date'] < cars['purchase_date']]) == 0

def add_claim_days(row):
    high_lim = row['days_purchase_to_censor'] - 1
    add_days = random.randint(1, high_lim)
    row['days_purchase_to_censor_fail'] = add_days
    end_date = row['purchase_date'] + datetime.timedelta(days=add_days)
    end_date = pd.to_datetime(end_date, format='%Y-%m-%d')
    row['censor_fail_date'] = end_date
    return(row)

fails = fails.apply(lambda row: add_claim_days(row), axis=1)

len(fails.loc[fails['censor_fail_date'] > fails['censor_date']]) == 0
fails = fails.loc[fails['censor_fail_date'] <= fails['censor_date']]
len(fails)

new_n_fails = fails.groupby('vin')['censor_fail_status'].apply(lambda x: (x=='F').sum()).reset_index(name='count')
max(new_n_fails['count'])
sum(new_n_fails['count'])
# Merge back failures and censors
fails_censors = pd.concat([fails, censors]).sort_values(by=['purchase_date', 'vin', 'censor_fail_date'], axis=0).reset_index(drop=True)
fails_censors.head(100)

# Now, for each event we need the days since prior event.
# For 1st failure, days_since_last = days from purchase date to 1st fail
# For 2nd - last failure, days_since_last = days from (i-1)th failure to this ith failure
# For censoring, days_since_last = days from purchase date to censor date for cars with no failures
#       and days_since_last = days from last failure to censor date for claims with failures.

# using a for loop for simplicity, but probably a better way to count events
# for each row, 1) check if vin is the same as the last row
#               2) if vin is the same as last row, let vin_event_number = last_row_vin_event_number + 1
#                   2a) if this_vin == last_vin & vin_event_number == 1 (after updating it):
#                       days_since_event is days from purchase_date to this row's fail_censor_date
#                       row['days_since_event'] = row['days_purchase_to_censor_fail']
#               3) if vin changed from last row, assign vin_event_number = 0

fails_censors['vin_event_number'] = None
fails_censors['days_since_event'] = None
last_vin = None
vin_event_number = 1
last_days = 0
for index, row in fails_censors.sort_values(by=['purchase_date', 'vin', 'censor_fail_date'], axis=0).iterrows():
    this_vin = row['vin']
    this_days = row['days_purchase_to_censor_fail']
    if this_vin == last_vin:
        vin_event_number = vin_event_number+1
        fails_censors.at[index, 'days_since_event'] = this_days - last_days  # subtract prior events days since purchase_date
    else: # this row new vin
        vin_event_number = 1
        fails_censors.at[index, 'days_since_event'] = this_days
    fails_censors.at[index, 'vin_event_number'] = vin_event_number
    last_vin = this_vin
    last_days = this_days

fails_censors.head(100)

fails_censors.to_csv("failures_censors_data.csv", header=True, index=False)
fails.to_csv("failures_data.csv", header=True, index=False)
