from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.api import VAR
from statsmodels.tools.eval_measures import aic
import pandas as pd
import Results.tools.helpers as hp
import numpy as np
import warnings
warnings.filterwarnings("ignore", message="'Q' is deprecated and will be removed in a future version, please use 'QE' instead.")
#TODO Finish commenting functions

def run_basic_VAR(data, lags, inputs):
    model_var = VAR(data)
    results = model_var.fit(lags)
    fitted_values = results.fittedvalues
    fitted_values_df = pd.DataFrame(fitted_values , columns=inputs, index=data.index)
    return fitted_values_df, data

def get_basic_VAR_predict(df, country, lags, inputs):
    country_data = hp.get_country(df, country)
    fitted_values, actual_values = run_basic_VAR(country_data, lags, inputs)
    return fitted_values, actual_values

def get_fulldata_RMSE(df, lags, variable, inputs):
    countries = df.index.get_level_values('Country').unique()
    total_squared_error = 0
    total_samples = 0
    
    for country in countries:
        fitted_values, actual_values = get_basic_VAR_predict(df, country, lags, inputs)
        # Gets mse
        mse = hp.calculate_mse(actual_values, fitted_values, variable, lags)
        # 
        total_squared_error += mse * len(actual_values[variable]) 
        total_samples += len(actual_values[variable])
        
        # print(f"{country} MSE: {mse}, Total Squared Error: {total_squared_error}")
    
    # Calculate final RMSE from accumulated squared errors
    total_mse = total_squared_error / total_samples
    rmse = np.sqrt(total_mse)
    
    return rmse


def run_VAR_predict(df, train, test, lags):
    '''runs the vector autoregression model on a train set and gets the predictions for a test dataframe.'''
    model_var = VAR(train)
    results = model_var.fit(lags) 
    predictions_var = results.forecast(train.values[-lags:], steps=len(test))
    time_periods = df.index.get_level_values('TIME_PERIOD')
    country = df.index.get_level_values('Country')[-1]
    time_index = pd.date_range(start=time_periods[-len(test)], periods=len(test), freq='Q')

    prediction_index = pd.MultiIndex.from_product([[country], time_index], names=['Country', 'TIME_PERIOD'])

    predictions_df = pd.DataFrame(predictions_var, columns=[test.columns], index=prediction_index)
    return predictions_df, test

def get_VAR_predict(df, train, test, country, lag):
    '''Gets the prediction for a vector autoregression on the test set of data.
    df: dataframe of stationary data
    train: dataframe of stationary training data
    test: dataframe of stationary test data
    country: Country you want to predict for
    lag: number of lags
    '''
    country_data = hp.get_country(df,country)
    country_train = hp.get_country(train, country)
    country_test = hp.get_country(test, country)
    var_prediction, actual = run_VAR_predict(country_data, country_train, country_test, lag)
    return var_prediction, actual


def get_test_errors(df, train, test, lags, variable):
    '''Gets the error metrics for a vector autoregression predictions by comparing it to the test set.
    df: dataframe of stationary data
    train: dataframe of stationary training data
    test: dataframe of stationary test data
    country: Country you want to predict for
    lag: number of lags
    '''
    countries = df.index.get_level_values('Country').unique()
    total_squared_error = 0
    total_absolute_error = 0 
    total_samples = 0
    for country in countries:
        fitted_values, actual_values = get_VAR_predict(df, train, test, country, lags)
        mse = hp.calculate_mse(actual_values, fitted_values, variable)
        total_squared_error += mse * len(actual_values[variable]) 
        mae = hp.calculate_mae(actual_values, fitted_values, variable)
        total_absolute_error += mae * len(actual_values[variable])
        total_samples += len(actual_values[variable])
        
    total_mse = total_squared_error / total_samples
    rmse = np.sqrt(total_mse)
    total_mae = total_absolute_error / total_samples #TODO check why this return was mae?
    return [total_mse, rmse, total_mae]


def retrend(detrend, trend, country):
    '''Retrends a countries data given the dataframe that stores the datas trend. 
    Meant for retrending VAR predictions data. 
    '''
    country_detrend = hp.get_country(detrend, country)
    country_trend = hp.get_country(trend, country)
    retrend_df = country_detrend.add(country_trend)
    retrend_df = retrend_df.dropna()
    return retrend_df


def aic_test(df):
    '''Performs the AIC test on the data. Helps deetermine the number of lags for a VAR.'''
    unique_countries = df.index.get_level_values('Country').unique()
    best_lag = []
    for country in unique_countries:
        country_data = hp.get_country(df, country)
        aic_scores = {}
        for lag_order in range(1, 9):  
            model = VAR(country_data)
            results = model.fit(lag_order)
            aic_scores[lag_order] = results.aic
        best_lag.append(min(aic_scores, key=aic_scores.get))

    return best_lag