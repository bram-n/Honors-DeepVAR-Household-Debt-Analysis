import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error

def get_country(df, country):
    '''Gets a specified countries data.
    df: panel dataframe
    country: country within specified dataframe'''
    country_data = df[df.index.get_level_values('Country') == country]
    return country_data




""" Splits each country's time series data into training and test sets, 
using the window of each country's data for training"""
def time_panel_split_predict(data, train_fraction=0.8):
    # Create empty DataFrames for training and testing
    train_data = pd.DataFrame()
    test_data = pd.DataFrame()

    countries = data.index.get_level_values('Country').unique()
    # Split data for each country
    for country in countries:
        # Get data for this country
        country_data = get_country(data, country).sort_values(by='TIME_PERIOD')
        
        # Calculate split point
        split_idx = int(len(country_data) * train_fraction)
        
        # Split country data and add to respective DataFrames
        train_data = pd.concat([train_data, country_data.iloc[:split_idx]])
        test_data = pd.concat([test_data, country_data.iloc[split_idx:]])
    return train_data, test_data


def get_test_data(df, train_fraction=0.8):
    '''Gets the test data for a specified dataframe and window. 
    Primarily used for visualization purposes'''
    split_idx = int(len(df) * train_fraction)
    test_data = df[split_idx:]
    return test_data


def exclude_country(df, country):
    newdf = df.copy()
    newdf = newdf.loc[newdf.index.get_level_values('Country') != country]
    return newdf


def create_model_comparison_latex_table(lstm_metrics, var_metrics, var_no_outlier_metrics):
    latex_table = r"""\begin{{table}}[htbp]
\centering
\caption{{Model Performance Comparison}}
\begin{{tabular}}{{lccc}}
\hline
\textbf{{Metric}} & \textbf{{DeepVAR}} & \textbf{{VAR}} & \textbf{{VAR (Excluding Growth Outliers)}} \\
\hline
Mean Squared Error (MSE) & {:.6f} & {:.6f} & {:.6f} \\
Root Mean Squared Error (RMSE) & {:.6f} & {:.6f} & {:.6f} \\
Mean Absolute Error (MAE) & {:.6f} & {:.6f} & {:.6f} \\
\hline
\end{{tabular}}
\end{{table}}""".format(
        lstm_metrics[0], var_metrics[0], var_no_outlier_metrics[0],
        lstm_metrics[1], var_metrics[1], var_no_outlier_metrics[1],
        lstm_metrics[2], var_metrics[2], var_no_outlier_metrics[2]
    )

    return latex_table


def calculate_mse(actual, predicted, variable, lags):
    actual_values_lagged = actual.iloc[lags:]
    fitted_values_lagged = predicted.iloc[lags:]
    
    actual_values = actual_values_lagged[variable].values
    fitted_values = fitted_values_lagged[variable].values

    if len(actual_values) != len(fitted_values):
        raise ValueError(f"Inconsistent lengths: actual ({len(actual_values)}) vs predicted ({len(fitted_values)})")

    squared_differences = (actual_values - fitted_values) ** 2
    mse = np.mean(squared_differences)
    
    return mse
    
    
def calculate_mae(actual, predicted, variable):
    mae = mean_absolute_error(actual[variable], predicted[variable])
    return mae

def calculate_percent_improvement(deepvar_metrics, compare_metrics):
    improvements = {}
    
    metric_names = ['total_mse', 'rmse', 'total_mae']
    
    for i, metric in enumerate(metric_names):
        deepvar_value = deepvar_metrics[i]
        compare_value = compare_metrics[i]
        
        if compare_value != 0:  
            improvement = ((compare_value - deepvar_value) / compare_value) * 100
            improvements[metric] = improvement
        else:
            improvements[metric] = None  
    
    return improvements

def visualize_model_performance(y_test, y_pred):
    plt.figure(figsize=(12, 6))

    plt.plot(y_test, label='True Values', color='skyblue', linestyle='-', linewidth=2)
    plt.plot(y_pred, label='Predicted Values', color='orange', linestyle='--', linewidth=2)

    plt.xlabel('Time', fontsize=14)
    plt.ylabel('Target', fontsize=14)
    plt.title('True vs. Predicted Values Over Time', fontsize=16)
    
    plt.legend(fontsize=12, loc='upper left')

    plt.tight_layout()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    plt.show()