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

X, y = chicago(return_X_y=True)

gam = PoissonGAM(s(0, n_splines=200) + te(3, 1) + s(2)).fit(X, y)

# Load and transform data into desired form
# pd.set_option('display.max_columns', None)
fails = pd.read_csv('failures_censors_data.csv').sort_values(by=['purchase_date', 'vin', 'days_purchase_to_censor_fail']).reset_index(drop=True)
cars = pd.read_csv('cars_data.csv').sort_values(by=['purchase_date', 'vin']).reset_index(drop=True)

dat = cars.merge(fails, on=['vin', 'purchase_date', 'n_fails', 'nvlw_end_date'], how='left').sort_values(
    by=['purchase_date', 'vin', 'days_purchase_to_censor_fail']).reset_index(drop=True)

dat['purchase_date'] = pd.to_datetime(dat['purchase_date'])
dat['cohort_year'] = dat['purchase_date'].apply(lambda x: x.strftime('%Y'))
dat['cohort_month'] = dat['purchase_date'].apply(lambda x: x.strftime('%m'))
dat['exposure_month'] = pd.to_datetime(dat['censor_fail_date']).apply(lambda x: x.strftime('%m'))
dat['years_purchase_to_censor_fail'] = dat['days_purchase_to_censor_fail']/365
dat['kmiles_purchase_to_censor_fail'] = dat['days_purchase_to_censor_fail']/1095 *dat['annual_mileage']*3/1000
dat['interval_start'] = pd.to_datetime(pd.to_datetime(dat['censor_fail_date']).apply(lambda x: x.strftime('%Y-%m-01')))
dat['interval_end'] = dat['interval_start'] + pd.DateOffset(months=1) + timedelta(days=-1)

def month_diff(end_dt, start_dt):
    end_dt = pd.to_datetime(end_dt)
    start_dt = pd.to_datetime(start_dt.strftime('%Y-%m-01'))
    rd = dateutil.relativedelta.relativedelta(end_dt, start_dt)
    months = rd.years*12 + rd.months
    return(months)

dat['duration'] = dat.apply(lambda x: month_diff(x.interval_start, x.purchase_date), axis=1)

# To get the month-by-month format with # active warranties each month and # failures each month, need to agg data
###  For every interval start/end, the # warranties at risk is all warranties with a start date
# First, our list of dates
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
vin_dat['purchase_date'] = pd.to_datetime(vin_dat['purchase_date'])
vin_dat['nvlw_end_date'] = pd.to_datetime(vin_dat['nvlw_end_date'])
int_vin_dat = pd.merge(vin_dat, int_dat, how ="cross")
def check_active (row):
    if row['purchase_date'] < row['interval_end'] and row['nvlw_end_date'] > row['interval_start']:
        return(True)
    else:
        return(False)

for index, row in int_vin_dat.iterrows():
    int_vin_dat.at[index, 'active_warranty'] = check_active(row)

# Summarize at the interval, model_year, car_type level
int_vin_dat.to_csv("vin_active_warranty.csv", header=True, index=False)

my_ct_active = int_vin_dat.groupby(['model_year', 'car_type', 'interval_start', 'interval_end']).agg(
    active_warranties = pd.NamedAgg(column='active_warranty', aggfunc='sum')).reset_index()
    # vins = pd.NamedAgg(column='vin', aggfunc=pd.Series.nunique)


def check_fails (dat):
    grouped = dat.groupby(['interval_start', 'interval_end', 'duration','exposure_month', 'car_type', 'model_year', 'censor_fail_status'])
    results = pd.DataFrame({'n_cf' : grouped.size()}).reset_index()
    results = pd.pivot(results, index = ['car_type', 'model_year', 'interval_start', 'interval_end', 'duration','exposure_month'],
                       columns='censor_fail_status', values='n_cf').reset_index().rename(
        columns={'C':'n_censored', 'F':'n_failed'})
    return(results)

mdat = check_fails(dat)
len(mdat)
len(my_ct_active)
mdat = pd.merge(mdat, my_ct_active, on = ['interval_start', 'interval_end', 'model_year', 'car_type'], how='left')
mdat.shape # (8154, 9)
mdat.to_csv("monthly_failures.csv", header=True, index=False)

sum(mdat.duplicated()) # No duplicates








# Let's try to fit the Poisson regression
datX = dat[['vin', 'model_year', 'car_type', 'annual_mileage']]

gam1 = PoissonGAM()


# Want predictor columns to be model_year, car_type

df = pd.DataFrame(data.data, columns=data.feature_names)[['mean radius', 'mean texture', 'mean perimeter', 'mean area','mean smoothness', 'mean compactness']]
target_df = pd.Series(data.target)
df.describe()


X = df[['mean radius', 'mean texture', 'mean perimeter', 'mean area','mean smoothness', 'mean compactness']]
y = target_df
#Fit a model with the default parameters
gam = LogisticGAM().fit(X, y)

gam.accuracy(X, y)

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