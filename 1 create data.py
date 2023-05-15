#########################################################
###              Reliability Analysis                 ###
###                   Emily Sheen                     ###
###        1) Create automobile contract data         ###
###        2) Create repair claims data               ###
#########################################################

### Assume we have a sample of 10,000 vehicles from model year 2020 with 36 month/36,000 mile bumper to bumper NVLW's.

import faker
import numpy as np
import pandas as pd
import datetime
from faker import Faker
fake = Faker()
import random
# fake.date_between(start_date='today', end_date = '+30y')

random.seed(7)
vin = [x for x in range(30000)]
manufacture_date = [fake.date_between(datetime.date(2019, 8, 1), datetime.date(2020,8,1)) for x in range(10000)] + \
            [fake.date_between(datetime.date(2020, 8, 1), datetime.date(2021,8,1)) for x in range(10000, 20000)] + \
            [fake.date_between(datetime.date(2021, 8, 1), datetime.date(2022,8,1)) for x in range(20000, 30000)]
def buy_date(dttime):
    add_days = random.randint(60, 360)
    return dttime + datetime.timedelta(days=add_days)

# Calling purchase_date method on each manufacture date to add a random number of days between 60 and 360
purchase_date = [buy_date(i) for i in manufacture_date]

model_year = [2020 for x in range(10000)] + [2021 for x in range(10000, 20000)] + [2022 for x in range(20000, 30000)]
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
    car_mileage = np.random.normal(avg, sd, 3000)
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



########################################################################################################################
######  2) REPAIR CLAIMS DATA:  Let's assume we were approached about problematic engine block claims leading to   #####
######     extensive NVLW claims.  We are interested in discovering the root cause of the engine block problems,   #####
######     which vehicle types and model years are afflicted with the issue, and whether a recall is necessary.    #####
######  *     Since for this example, we are designing the dataset to be explored, we will assume this is a 2021   #####
######     and 2022 problem for trucks and heavy duty trucks.  We will assume Poisson distribution with lambda = 3 #####
######     for 2021 and 2022 trucks and lambda = 4 for 21/22 heavy duty trucks. Let lambda values vary for other   #####
######     vehicle types and model years, between 0.1 and 0.5.                                                     #####
######  *     Let lambda = 3 for Engine Block claims w/in the 3 yr / 36K mi NVLW period, both trucks MY 2021/22    #####
######  *     Let lambda < 1 for all other car types, all 3 model years                                            #####
########################################################################################################################

# Let's generate a random number of engine block claims for each vehicle in our sample
# Set lambda values for each car type and model year

lambdas = pd.DataFrame({"car_type": cars['car_type'].unique().tolist() * 3,
                        "model_year": [2020]*10 + [2021]*10 + [2022]*10,
                        "lambda": [0.1, 0.2, 0.3, 0.1, 0.5, 0.1, 0.2, 0.2, 0.1, 0.01,
                                   0.01, 0.3,  3, 0.1,  3,  0.2, 0.3, 0.3, 0.1, 0.01,
                                   0.01, 0.2,  3, 0.2,  3,  0.2, 0.2, 0.2, 0.1, 0.01]})
print(lambdas)

print(cars.groupby(['car_type', 'model_year']).agg(
    avg_mileage = pd.NamedAgg(column='annual_mileage', aggfunc=np.mean),
    sd_mileage = pd.NamedAgg(column='annual_mileage', aggfunc=np.std),
    n_cars = pd.NamedAgg(column='vin', aggfunc='count')
))


def get_claim_counts(carty, my):
    lam = lambdas.loc[(lambdas['car_type'] == carty) & (lambdas['model_year'] == my), 'lambda']
    n_claims = np.random.poisson(lam, size=1000)
    return n_claims

get_claim_counts('Sedan', 2020)

for index, row in lambdas.iterrows():
    cart = row['car_type']
    my = row['model_year']
    cars.loc[(cars['car_type'] == cart) & (cars['model_year'] == my), 'n_claims'] = get_claim_counts(cart, my)

# n_claims will occur within the NVLW period of 3 years / 36,000 miles
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

#### Now we need to determine which day in the NVLW period the claims will occur.
# We will assume the claims can occur in [purchase_date, nvlw_end_date], but then censor back out
# any claims in [today(), nvlw_end_date] for newer vehicles


claims = cars[["vin", "purchase_date", 'nvlw_end_date', "n_claims"]]
               # "manufacture_date",  "model_year", "car_type",
               # "annual_mileage", 'nvlw_timeout', 'days_to_mileout', 'nvlw_mileout',

claims = claims[claims['n_claims'] >= 1]
# Make duplicates for every n_claims > 1
claims = claims.loc[claims.index.repeat(claims.n_claims)]

len(claims)

claims['days_to_claim'] = [random.randint(0, 1095) for i in range(0, len(claims))]

def add_claim_days (row):
    add_days = round(row['days_to_claim'])
    end_date = row['purchase_date'] + datetime.timedelta(days=add_days)
    end_date = pd.to_datetime(end_date, format='%Y-%m-%d')
    return(end_date)

claims['claim_date'] = claims.apply(lambda row: add_claim_days(row), axis=1)

future_claims = claims.loc[claims['claim_date'] > pd.to_datetime(datetime.date.today(), format='%Y-%m-%d')]
claims = claims.loc[claims['claim_date'] <= pd.to_datetime(datetime.date.today(), format='%Y-%m-%d')]
len(future_claims)
len(claims)

claims.to_csv("claims_data.csv", header=True, index=False)
future_claims.to_csv("future_claims_data.csv", header=True, index=False)