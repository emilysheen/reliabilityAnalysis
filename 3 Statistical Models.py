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

dat.head()

###  For every interval start/end, the # warranties at risk is all warranties with a start date





# For grouped data by purchase cohort,
print(dat.groupby(['cohort_year', 'cohort_month', 'interval_start', 'interval_end', 'duration', 'exposure_month']).agg(
    new_warranties =pd.NamedAgg(column='annual_mileage', aggfunc=np.mean),
    avg_mileage = pd.NamedAgg(column='annual_mileage', aggfunc=np.mean),
    sd_mileage = pd.NamedAgg(column='annual_mileage', aggfunc=np.std),
    n_cars = pd.NamedAgg(column='vin', aggfunc='count')
))

def check_warranties (dat):
    grouped = dat.groupby(['cohort_year', 'cohort_month', 'interval_start', 'interval_end', 'duration','exposure_month'])
    min_date = row['interval_start']
    max_date = row['interval_end']
    num_warranties = len(dat.loc[dat['purchase_date'] < min_date and dat['nvlw_end_date'] > max_date])
    return(num_warranties)

def add_days (row):
    mileout_days = round(row['days_to_mileout'])
    mileout = row['purchase_date'] + datetime.timedelta(days=mileout_days)
    mileout = pd.to_datetime(mileout, format='%Y-%m-%d')
    return(mileout)

dat.groupby(['cohort_year', 'cohort_month', 'interval_start', 'interval_end', 'exposure_month']).agg(

)



pivot = dat.pivot_table(columns= ['cohort_year', 'cohort_month', 'interval_start', 'interval_end', 'censor_fail_status'],
                        values= ['']
                        aggfunc ='count')
print (pivot)



# To get the month-by-month format with # active warranties each month and # failures each month, need to agg data
# sd = "2018-10-01"
# start_date = datetime.strptime(sd, "%Y-%m-%d")

day_dat = dat.groupby(['cohort_year', 'cohort_month', ])


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