import numpy as np
import matplotlib.pyplot as plt
import helpers as hp
import pandas as pd
def create_coefficient_matrices(coef_df):
    """
    Create coefficient matrices from the panel VAR coefficient DataFrame
    
    Parameters:
    coef_df: DataFrame with coefficients in the format provided
    
    Returns:
    list of numpy arrays, each representing a lag's coefficient matrix
    """
    n_vars = 3  
    n_lags = len(coef_df) // n_vars 
    
    coeff_matrices = []
    
    for lag in range(1, n_lags + 1):
        lag_matrix = np.zeros((n_vars, n_vars))

        start_idx = (lag - 1) * n_vars
        lag_coeffs = coef_df.iloc[start_idx:start_idx + n_vars]
        
        # Fill the matrix with coefficients
        lag_matrix[0, :] = lag_coeffs['GDP_eq']  # GDP equation coefficients
        lag_matrix[1, :] = lag_coeffs['HD_eq']   # Household Debt equation coefficients
        lag_matrix[2, :] = lag_coeffs['PD_eq']   # Private Debt equation coefficients
        
        coeff_matrices.append(lag_matrix)
    
    return coeff_matrices

def compute_irf(coeff_matrices, periods=10, shock_var=0, shock_size=1.0):
    """
    Compute Impulse Response Function
    
    Parameters:
    coeff_matrices: list of coefficient matrices for each lag
    periods: number of periods to forecast
    shock_var: index of variable to shock (0=GDP, 1=HD, 2=PD)
    shock_size: size of the shock
    
    Returns:
    irf_result: array of shape (periods, n_vars) containing IRF values
    """
    n_vars = coeff_matrices[0].shape[0]
    irf = np.zeros((periods, n_vars))
    
    # Initial shock
    shock = np.zeros(n_vars)
    shock[shock_var] = shock_size
    irf[0] = shock
    
    # Compute IRF
    for t in range(1, periods):
        for lag, coeff_matrix in enumerate(coeff_matrices, 1):
            if t >= lag:
                irf[t] += coeff_matrix @ irf[t-lag]
    
    return irf

def run_panel_VAR_predict(df, test, lags, coeff_matrices):
    """
    Run panel VAR predictions using pre-estimated coefficient matrices
    
    Parameters:
    df: DataFrame containing the data for initial values
    test: test dataset to predict for
    lags: number of lags
    coeff_matrices: list of coefficient matrices
    
    Returns:
    predictions_df: DataFrame with predictions
    test: actual test values
    """
    # Get the last lags periods of data before test period
    initial_values = df[df.index.get_level_values('TIME_PERIOD') < 
                       test.index.get_level_values('TIME_PERIOD')[0]].tail(lags).values
    
    # Number of periods to forecast
    n_periods = len(test)
    n_vars = coeff_matrices[0].shape[0]
    forecast = np.zeros((n_periods, n_vars))
    
    # Generate forecasts
    for t in range(n_periods):
        if t < lags:
            last_values = np.vstack([initial_values[-(lags-t):], forecast[:t]])
        else:
            last_values = forecast[t-lags:t]
            
        # Calculate forecast for time t
        forecast_t = np.zeros(n_vars)
        for lag, coeff_matrix in enumerate(coeff_matrices):
            forecast_t += coeff_matrix @ last_values[lag]
        forecast[t] = forecast_t
    
    # Create DataFrame with predictions
    time_periods = test.index.get_level_values('TIME_PERIOD')
    country = test.index.get_level_values('Country')[0]
    prediction_index = pd.MultiIndex.from_product([[country], time_periods], 
                                                names=['Country', 'TIME_PERIOD'])
    predictions_df = pd.DataFrame(forecast, columns=test.columns, index=prediction_index)
    
    return predictions_df, test

def get_panel_VAR_predict(df, test, country, lags, coeff_matrices):
    """
    Get panel VAR predictions for a specific country
    
    Parameters:
    df: full dataset
    test: test dataset
    country: country to predict for
    lags: number of lags
    coeff_matrices: list of coefficient matrices
    
    Returns:
    predictions: predicted values
    actual: actual values
    """
    country_data = hp.get_country(df, country)
    country_test = hp.get_country(test, country)
    
    predictions, actual = run_panel_VAR_predict(country_data, country_test, 
                                              lags, coeff_matrices)
    return predictions, actual


def plot_irf(irf_result, variable_names=['GDP', 'Household Debt', 'Private Debt'], 
             shock_var=0, shock_size=1.0, figsize=(10, 12)):
    """
    Plot the Impulse Response Functions
    
    Parameters:
    irf_result: array of IRF values
    variable_names: list of variable names
    shock_var: index of shocked variable
    shock_size: size of the shock
    figsize: figure size tuple
    """
    periods = len(irf_result)
    fig, axes = plt.subplots(len(variable_names), 1, figsize=figsize)
    
    shock_description = f'{shock_size} unit'
    fig.suptitle(f'Impulse Response Functions to {shock_description} {variable_names[shock_var]} Shock')
    
    if len(variable_names) == 1:
        axes = [axes]
    
    for i, (ax, var_name) in enumerate(zip(axes, variable_names)):
        ax.plot(range(periods), irf_result[:, i], 'b-', label=f'Response of {var_name}')
        ax.axhline(y=0, color='r', linestyle='--')
        ax.set_xlabel('Periods')
        ax.set_ylabel('Response')
        ax.legend()
        ax.grid(True)
    
    plt.tight_layout()
    return fig

def generate_irf_from_coefficients(coef_df, shock_var=1, shock_size=1.0, periods=10):
    """
    Generate IRF from coefficient DataFrame
    
    Parameters:
    coef_df: DataFrame containing VAR coefficients
    shock_var: which variable to shock (0=GDP, 1=HD, 2=PD)
    shock_size: size of the shock
    periods: number of periods to forecast
    
    Returns:
    fig: matplotlib figure object
    irf_result: array of IRF values
    """
    coeff_matrices = create_coefficient_matrices(coef_df)
    
    #IRF
    irf_result = compute_irf(coeff_matrices, 
                           periods=periods, 
                           shock_var=shock_var, 
                           shock_size=shock_size)
    
    fig = plot_irf(irf_result, 
                  shock_var=shock_var, 
                  shock_size=shock_size)
    
    return fig, irf_result


