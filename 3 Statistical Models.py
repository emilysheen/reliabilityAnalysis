#########################################################
###              Reliability Analysis                 ###
###                   Emily Sheen                     ###
###  Survival Functions, Distribution Fitting, and    ###
###         Failure Rate Exploratory Analysis         ###
#########################################################

import pandas as pd
import pandasql as ps
from pygam import PoissonGAM, s, te
from pygam.datasets import chicago
from datetime import datetime, timedelta
import dateutil
import numpy as np

# Load and transform data into desired monthly form
# pd.set_option('display.max_columns', None)
fails = pd.read_csv('failures_censors_data.csv').sort_values(by=['purchase_date', 'vin', 'days_purchase_to_censor_fail']).reset_index(drop=True)
cars = pd.read_csv('cars_data.csv').sort_values(by=['purchase_date', 'vin']).reset_index(drop=True)

dat = cars.merge(fails, on=['vin', 'purchase_date', 'n_fails', 'nvlw_end_date'], how='left').sort_values(
    by=['purchase_date', 'vin', 'days_purchase_to_censor_fail']).reset_index(drop=True)  #46,135

dat['purchase_date'] = pd.to_datetime(dat['purchase_date'])
dat['cohort_year'] = dat['purchase_date'].apply(lambda x: x.strftime('%Y'))
dat['cohort_month'] = dat['purchase_date'].apply(lambda x: x.strftime('%m'))
dat['exposure_month'] = pd.to_datetime(dat['censor_fail_date']).apply(lambda x: x.strftime('%m'))
dat['years_purchase_to_censor_fail'] = dat['days_purchase_to_censor_fail']/365
dat['kmiles_purchase_to_censor_fail'] = dat['days_purchase_to_censor_fail']/1095 *dat['annual_mileage']*3/1000
dat['interval_start'] = pd.to_datetime(pd.to_datetime(dat['censor_fail_date']).apply(lambda x: x.strftime('%Y-%m-01')))
dat['interval_end'] = dat['interval_start'] + pd.DateOffset(months=1) + timedelta(days=-1)
# Still 46,135

def month_diff(end_dt, start_dt):
    end_dt = pd.to_datetime(end_dt)
    start_dt = pd.to_datetime(start_dt.strftime('%Y-%m-01'))
    rd = dateutil.relativedelta.relativedelta(end_dt, start_dt)
    months = rd.years*12 + rd.months
    return(months)

dat['duration'] = dat.apply(lambda x: month_diff(x.interval_start, x.purchase_date), axis=1) # 46,135

# Let's split by VIN into TRAINING AND TESTING sets
test_dat = dat[['vin', 'car_type', 'model_year']].drop_duplicates(subset=['vin', 'car_type', 'model_year'], keep='first')\
    .groupby(['car_type', 'model_year']).apply(pd.DataFrame.sample, n=200, replace=False).reset_index(drop=True)
test_dat['train_test'] = 'Test'  # 6,000

dat = dat.merge(test_dat, on = ['vin', 'car_type', 'model_year'], how='left') # Still 46,135
dat['train_test'] = dat['train_test'].fillna('Train')

dat[['train_test']].groupby('train_test').size()
# Test     9267
# Train    36868
vin_test = dat[['vin', 'train_test']].drop_duplicates(keep = 'first')
vin_test.to_csv("vin_train_test_table.csv", header=True, index=False)

# To get the month-by-month format with # active warranties each month and # failures each month, need to agg data
###  For every interval start/end, the # warranties at risk is all warranties with a start date
# START HERE
min(dat['interval_start'])
min(dat['purchase_date']) # min is 2018-10-05, starting interval is 2018-10-01 to 2018-10-31

cutoff_dt = max(pd.to_datetime(dat['censor_fail_date']))
cur_dt = pd.to_datetime("2018-10-01")
int_starts = []
int_ends = []
warranties_at_risk = []
while cur_dt < cutoff_dt:
    int_starts.append(cur_dt) # Add cur_dt to int_starts list
    end_dt = cur_dt + pd.DateOffset(months=1) + timedelta(days=-1)# Compute int_ends
    int_ends.append(end_dt)
    cur_dt = cur_dt + pd.DateOffset(months=1)

int_dat = pd.DataFrame({'interval_start' : int_starts, 'interval_end' : int_ends})
vin_dat = cars[['vin', 'purchase_date', 'nvlw_end_date', 'model_year', 'car_type']]
vin_dat[["purchase_date", "nvlw_end_date"]] = vin_dat[["purchase_date", "nvlw_end_date"]].apply(pd.to_datetime) # SettingWithCopyWarning

# vin_dat.loc[:, ("purchase_date", "nvlw_end_date")] = vin_dat.loc[:, ("purchase_date", "nvlw_end_date")].copy().apply(pd.to_datetime)
# Above command gives SettingWithCopyWarning and DeprecationWarning: In a future version, `df.iloc[:, i] = newvals` will attempt to set the values inplace instead of always setting a new array.
# vin_dat['purchase_date'] = vin_dat['purchase_date'].apply(pd.to_datetime) # SettingWithCopyWarning
# vin_dat['purchase_date'] = pd.to_datetime(vin_dat['purchase_date']) # SettingWithCopyWarning
# vin_dat['nvlw_end_date'] = pd.to_datetime(vin_dat['nvlw_end_date']) # SettingWithCopyWarning
int_vin_dat = pd.merge(vin_dat, int_dat, how ="cross")
def check_active (row):
    if row['purchase_date'] < row['interval_end'] and row['nvlw_end_date'] > row['interval_start']:
        return(True)
    else:
        return(False)

for index, row in int_vin_dat.iterrows():
    int_vin_dat.at[index, 'active_warranty'] = check_active(row) # 12:38 to 12:40


# int_vin_dat = pd.read_csv("vin_active_warranty.csv")  # 1680000
# vin_test = pd.read_csv("vin_train_test_table.csv")
int_vin_dat = int_vin_dat.merge(vin_test, on='vin', how='left')

# int_vin_dat.to_csv("vin_active_warranty.csv", header=True, index=False)   # before added train/test variable
# int_vin_dat.to_csv("vin_active_warranty_train_test.csv", header=True, index=False)  # after adding train/test

# Summarize at the interval, model_year, car_type, and train_test level
my_ct_active = int_vin_dat.groupby(['model_year', 'car_type', 'train_test', 'interval_start', 'interval_end']).agg(
    active_warranties = pd.NamedAgg(column='active_warranty', aggfunc='sum')).reset_index()
    # vins = pd.NamedAgg(column='vin', aggfunc=pd.Series.nunique)


def check_fails (dat):
    grouped = dat.groupby(['interval_start', 'interval_end', 'duration','exposure_month', 'car_type', 'model_year', 'train_test', 'censor_fail_status'])
    results = pd.DataFrame({'n_cf' : grouped.size()}).reset_index()
    results = pd.pivot(results, index = ['train_test', 'car_type', 'model_year', 'interval_start', 'interval_end', 'duration','exposure_month'],
                       columns='censor_fail_status', values='n_cf').reset_index().rename(
        columns={'C':'n_censored', 'F':'n_failed'})
    return(results)

mdat = check_fails(dat)
len(mdat)  #11277
len(my_ct_active)  #3360
vin_dat[["purchase_date", "nvlw_end_date"]] = vin_dat[["purchase_date", "nvlw_end_date"]].apply(pd.to_datetime) # SettingWithCopyWarning

my_ct_active[['interval_start', 'interval_end']] = my_ct_active[['interval_start', 'interval_end']].apply(pd.to_datetime)
mdat = pd.merge(mdat, my_ct_active, on=['interval_start', 'interval_end', 'model_year', 'car_type', 'train_test'], how='left')
mdat.shape # (11277, 10)

sum(mdat.duplicated()) # No duplicates
sum(mdat['train_test'].isna()) # 0
mdat = mdat.fillna(0) # fill all the na values with 0, replace NaN values with 0, specifically for n_failed and n_censored
mdat.head(100)
mdat['ln_warranties'] = np.log(mdat['active_warranties'])

# Must make a dictionary key for car_type to int values
cart_dict = dict({'Compact SUV': 1, 'Convertible': 2, 'Electric Vehicle':3, 'Hatchback':4,
                  'Large SUV':5, 'Mid-Size SUV':6, 'Minivan':7,
                  'Sedan':8, 'Truck':9, 'Heavy Duty Truck':10})
mdat['car_type_code'] = mdat["car_type"].apply(lambda x: cart_dict.get(x))


mdat.to_csv("monthly_failures_2023May24.csv", header=True, index=False)

#########################################################################################
### Perform Poisson GAM Model with pygam
### Load the previously saved month-by-month data
    # In pyGAM, we specify the functional form using terms:
    #
    # l() linear terms: for terms like 𝑋𝑖
    # s() spline terms
    # f() factor terms
    # te() tensor products
    # intercept

# prior dataset is 'monthly_failures.csv'
mdat = pd.read_csv('monthly_failures_2023May24.csv')
mdat.columns # ['train_test', 'car_type', 'model_year', 'interval_start', 'interval_end', 'duration',
             #  'exposure_month', 'n_censored', 'n_failed', 'active_warranties', 'ln_warranties', 'car_type_code']


# let's remove car_type and see if the model runs
trainX = mdat.loc[mdat['train_test'] == 'Train', ['model_year', 'car_type_code', 'ln_warranties', 'duration', 'exposure_month']].reset_index(drop = True).to_numpy()
trainy = mdat.loc[mdat['train_test'] == 'Train', 'n_failed'].reset_index(drop = True).to_numpy()
testX = mdat.loc[mdat['train_test'] == 'Test', ['model_year', 'car_type_code', 'ln_warranties', 'duration', 'exposure_month']].reset_index(drop = True).to_numpy()
testy = mdat.loc[mdat['train_test'] == 'Test', 'n_failed'].reset_index(drop = True).to_numpy()


from pygam import PoissonGAM, s, f, l, te
from pygam.datasets import wage, chicago, default
# X, y = chicago(return_X_y=True)  # sample dataset
# X, y = wage(return_X_y=True)     # sample dataset
# X, y = default(return_X_y=True)  # sample dataset


# First GAM: Splines for duration and exposure month, factors for model_year and car_type, linear for ln_warraties
# Before fitting the regression we suspect this example will not exhibit seasonality due to how the failure dates
# were randomly generated.   # f(0) + f(1) + l(2) + s(3) + s(4)
gam1 = PoissonGAM(f(0) + f(1) + l(2) + s(3) + s(4), fit_intercept=True).fit(trainX, trainy)  # Remove f(1) for car_type

# Let's predict on our test set
gam1_predy = gam1.predict(testX, testy)

# Want predictor columns to be model_year, car_type




XX = generate_X_grid(gam)
plt.rcParams['figure.figsize'] = (28, 8)
fig, axs = plt.subplots(1, len(data.feature_names[0:6]))
titles = data.feature_names
for i, ax in enumerate(axs):
    pdep, confi = gam.partial_dependence(XX, feature=i+1, width=.95)
    ax.plot(XX[:, i], pdep)
    ax.plot(XX[:, i], confi[0][:, 0], c='grey', ls='--')
    ax.plot(XX[:, i], confi[0][:, 1], c='grey', ls='--')
    ax.set_title(titles[i])
plt.show()