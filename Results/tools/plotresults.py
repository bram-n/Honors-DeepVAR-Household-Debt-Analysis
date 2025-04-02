import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import Results.tools.helpers as hp
import Results.var.var as var
import Results.deepvar.lstm as lstm
import Results.deepvar.deepvar as deepvar

def plot_country_comparison(country, variable, inputs_lstm, inputs_var, lags, var_data, all_lstm_data, dict):
    country_data = hp.get_country(all_lstm_data, country)
    country_VAR, _ = var.get_basic_VAR_predict(var_data, country, lags, inputs_var)
    lstm_predict = lstm.fill_forecast_values(country_data, inputs_lstm, variable, dict)

    plt.figure(figsize=(15, 6))

    plt.plot(country_data.index.get_level_values('TIME_PERIOD'), country_data[variable], label=f"{country} Actual", color='skyblue')
    plt.plot(country_VAR.index.get_level_values('TIME_PERIOD'), country_VAR[variable], label=f"{country} VAR", color='green', linestyle='--')
    plt.plot(lstm_predict.index.get_level_values('TIME_PERIOD'), lstm_predict[variable], label=f"{country} LSTM Forecast", color='orange', linestyle='--')

    plt.title(f'log_GDP Predictions for {country} Compared to Actual')
    plt.xlabel('Date')
    plt.ylabel(f'{variable}')
    
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    plt.show()


def plot_country_predictions_test(country, variable, inputs, lags, df, dict, train_dt_var, test_dt_var, final_detrend, panel_predictions, cross_sectional_means, train_fraction=0.8):
    country_data = hp.get_country(df, country)
    country_var_detrend_predict, _ = var.get_VAR_predict(final_detrend, train_dt_var, test_dt_var, country, lags)

    # LSTM predictions 
    country_test_data = hp.get_test_data(country_data, train_fraction=train_fraction)
    lstm_predict = lstm.fill_forecast_values(country_test_data, inputs, variable, dict)

    lstm_predict = lstm_predict + cross_sectional_means
    # Get panel VAR predictions and add back means
    panel_predictions_country = hp.get_country(panel_predictions, country)
    panel_predictions_adjusted = panel_predictions_country.copy()
    panel_predictions_adjusted[variable] = panel_predictions_adjusted[variable] + cross_sectional_means[variable]

    # Deep var predictions
    steps_to_predict = len(country_test_data)
    deepvar_predict_dict = deepvar.autoregressive_predict(country_test_data, inputs, steps_to_predict, dict, lags)
    deepvar_predict = deepvar_predict_dict[variable]
    
    deepvar_predict_df = pd.DataFrame(
        deepvar_predict, 
        index=country_test_data.index,
        columns=[variable]
    )
    
    deepvar_predict_df[variable] = deepvar_predict_df[variable] + cross_sectional_means[variable]

    plt.figure(figsize = (12,6))

    plt.plot(country_data.index.get_level_values('TIME_PERIOD'), country_data[variable], label=f"{country} Actual", color='skyblue')
    plt.plot(country_var_detrend_predict.index.get_level_values('TIME_PERIOD'), country_var_detrend_predict[variable], label=f"{country} VAR Predicted", color='green', linestyle='--')
    plt.plot(lstm_predict.index.get_level_values('TIME_PERIOD'), lstm_predict[variable], label=f"{country} LSTM Forecast", color='orange', linestyle='--')
    plt.plot(deepvar_predict_df.index.get_level_values('TIME_PERIOD'), deepvar_predict_df[variable], label=f"{country} Deep VAR Forecast", color='red', linestyle='--')
    plt.plot(panel_predictions_adjusted.index.get_level_values('TIME_PERIOD'),
              panel_predictions_adjusted[variable], label='Panel VAR', color='purple')
    
    plt.title(f'{variable} Predictions for {country} Compared to Actual')
    plt.xlabel('Date')
    plt.ylabel(f'{variable}')
    
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    plt.show()


    

