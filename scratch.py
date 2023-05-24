import numpy as np
import pandas as pd
np.random.seed(2017)
N = 10000
df = pd.DataFrame({
    'Buy/Sell': np.random.randint(2, size=N),
    'Trader': np.random.randint(1000, size=N)})

def using_select(df):
    grouped = df.groupby(['Trader'])
    result = grouped['Buy/Sell'].agg(['sum', 'count'])
    means = grouped['Buy/Sell'].mean()
    result['Buy/Sell'] = np.select(condlist=[means>0.5, means<0.5], choicelist=[1, 0],
        default=np.nan)
    return result

def categorize(x):
    m = x.mean()
    return 1 if m > 0.5 else 0 if m < 0.5 else np.nan

def using_custom_function(df):
    result = df.groupby(['Trader'])['Buy/Sell'].agg([categorize, 'sum', 'count'])
    result = result.rename(columns={'categorize' : 'Buy/Sell'})
    return result




import pandas as pd

vin_dat = pd.DataFrame({'vin' : [1, 2, 3, 4, 5],
    'purchase_date' : ["2020-03-26", "2021-04-05", "2021-12-17", "2021-12-18", "2022-01-30"],
    'nvlw_end_date' : ["2023-03-26", "2024-04-05", "2024-12-17", "2024-12-18", "2025-01-30"] })

vin_dat.loc[:, ("purchase_date", "nvlw_end_date")] = vin_dat.loc[:, ("purchase_date", "nvlw_end_date")].copy().apply(pd.to_datetime)
# DeprecationWarning: In a future version, `df.iloc[:, i] = newvals` will attempt to set the values inplace instead of always setting a new array.

vin_dat[["purchase_date", "nvlw_end_date"]] = vin_dat[["purchase_date", "nvlw_end_date"]].apply(pd.to_datetime)
# This works without an error.

vin_dat['purchase_date'] = vin_dat['purchase_date'].apply(pd.to_datetime) # No error
vin_dat['purchase_date'] = pd.to_datetime(vin_dat['purchase_date']) # No error
vin_dat['nvlw_end_date'] = pd.to_datetime(vin_dat['nvlw_end_date']) # No error