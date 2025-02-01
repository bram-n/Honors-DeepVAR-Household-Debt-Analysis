from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.api import VAR
import pandas as pd
import numpy as np
import helpers as hp

def adf_test(df):
    '''Performs the Augmented Dickey-Fuller test which tests 
    for stationarity in a time series.
    df: takes in a given time series datafram
    returns True if the data is stationary and False if it is not stationary.'''
    if df.nunique() == 1 or df.isnull().all():
        print(f"Skipping ADF test for constant or empty series: {df.name}")
        return False
    result = adfuller(df)
    if result[1] <= 0.05:
        # print("Series is stationary")
        return True
    else:
        # print("Series is not stationary")
        return False
    
def difference(df, periods):
    return df.diff(periods=periods).dropna() 
        
'''Detrends data if it doesn't pass the ADF test. Detrends by country. 
Stores trend in a dataframe for visualization purposes.'''
def detrend_data(df, inputs, periods):
    final_detrend = pd.DataFrame()
    final_trend = pd.DataFrame()
    unique_countries = df.index.get_level_values('Country').unique()
    print(unique_countries)
    
    for country in unique_countries:
        country_data = hp.get_country(df, country)
        country_detrend = country_data.copy()
        for column in inputs:
            # if column in country_data.columns:
            if column != 'policy_rate':
                country_detrend[column] = difference(country_data[column], periods)
        final_detrend = pd.concat([final_detrend, country_detrend], axis=0)
        final_detrend = final_detrend.dropna()
    return final_detrend



