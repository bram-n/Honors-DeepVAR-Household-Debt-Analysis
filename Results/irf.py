import torch
import numpy as np
import deepvar
import pandas as pd
import matplotlib.pyplot as plt

# TODO comment this
def predict(df, inputs, variable_to_shock, shock_value, variable, steps_to_predict, dict, LAGS):
    for model, _, _, _, _ in dict.values():
        model.eval()
    input_data = df.iloc[0][inputs].to_frame().T
    current_input = input_data.to_numpy().reshape(1, -1)
    
    baseline_predictions = {key: [] for key in dict.keys()}
    with torch.no_grad():
        for _ in range(steps_to_predict):
            new_row = {}
            
            for var, (model, _, _, scaler_X, scaler_y) in dict.items():
                prediction = deepvar.predict_next(model, scaler_X, scaler_y, current_input)
                
                baseline_predictions[var].append(prediction)

                new_row[f'{var}_lag1'] = prediction
                for lag in range(2, LAGS):
                    new_row[f'{var}_lag{lag}'] = input_data[f'{var}_lag{lag-1}'].values[0]
            
            # Update input data for next iteration
            input_data = pd.DataFrame([new_row])
            current_input = input_data.iloc[0].to_numpy().reshape(1, -1)
    
    # Shock
    shocked_input_data = df.iloc[0][inputs].to_frame().T
    shocked_input_data.iloc[0, shocked_input_data.columns.get_loc(variable_to_shock)] += shock_value
    current_shocked_input = shocked_input_data.to_numpy().reshape(1, -1)
    
    # Initialize predictions dictionary for shocked scenario
    shocked_predictions = {key: [] for key in dict.keys()}
    with torch.no_grad():
        for _ in range(steps_to_predict):
            new_shocked_row = {}
            
            for var, (model, _,_, scaler_X, scaler_y) in dict.items():
                shocked_prediction = deepvar.predict_next(model, scaler_X, scaler_y, current_shocked_input)
                
                shocked_predictions[var].append(shocked_prediction)
                
                new_shocked_row[f'{var}_lag1'] = shocked_prediction
                for lag in range(2, 4):
                    new_shocked_row[f'{var}_lag{lag}'] = shocked_input_data[f'{var}_lag{lag-1}'].values[0]
            
            # Update input data for next iteration
            shocked_input_data = pd.DataFrame([new_shocked_row])
            current_shocked_input = shocked_input_data.iloc[0].to_numpy().reshape(1, -1)
    

    irf_results = [0] 
    for i in range(steps_to_predict):
        difference = shocked_predictions[variable][i] - baseline_predictions[variable][i]
        irf_results.append(difference)
    return irf_results

def plot(irf_results, variable, title, y_label):    
    plt.figure(figsize=(10, 6))
    plt.plot(irf_results, label=variable)
    
    plt.title(f'{title}')
    plt.xlabel('Quarters')
    plt.ylabel(f'{y_label}')
    plt.legend()
    plt.grid(True)
    plt.show()