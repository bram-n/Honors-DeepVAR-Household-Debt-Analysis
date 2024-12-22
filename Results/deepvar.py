import torch
import numpy as np
import pandas as pd
import lstm
import helpers as hp


def predict_next(model, scaler_X, scaler_y, current_input):
    scaled_input = scaler_X.transform(current_input)
    input_tensor = torch.FloatTensor(scaled_input).unsqueeze(0)
    
    prediction = model(input_tensor)
    
    actual_prediction = scaler_y.inverse_transform(prediction.numpy().reshape(1, -1))[0][0]
    return actual_prediction
    
def autoregressive_predict(df, inputs, steps_to_predict, dict):
    lstm.set_seed()
    # Evaluate all models
    for model, _, _ in dict.values():
        model.eval()
    
    # Prepare initial input
    input_data = df.iloc[0][inputs].to_frame().T
    current_input = input_data.to_numpy().reshape(1, -1)
    
    # Initialize predictions dictionary
    predictions_dict = {key: [] for key in dict.keys()}
    
    with torch.no_grad(): #no grad since predicting
        for _ in range(steps_to_predict):
            new_row = {}
            
            for variable, (model, scaler_X, scaler_y) in dict.items():
                # Predict for each variable
                prediction = predict_next(model, scaler_X, scaler_y, current_input)
                
                # Store prediction
                predictions_dict[variable].append(prediction)
                
                # Prepare new row with lagged values
                new_row[f'{variable}_lag1'] = prediction
                for lag in range(2, 4):
                    new_row[f'{variable}_lag{lag}'] = input_data[f'{variable}_lag{lag-1}'].values[0]
            # Update input data for next iteration
            input_data = pd.DataFrame([new_row])
            current_input = input_data.iloc[0].to_numpy().reshape(1, -1)
    
    return predictions_dict

def test_errors(df, test, variable, dict):
    lstm.set_seed()
    
    countries = df.index.get_level_values('Country').unique()
    df = df.drop(columns = {'log_pd', 'log_hhd', 'log_CPI', 'log_GDP'})
    total_squared_error = 0
    total_absolute_error = 0 
    total_samples = 0
    inputs = hp.get_unlagged_variables(df)
    for country in countries:
        # country_data = hp.get_country(df, country)
        actual_values = hp.get_country(test, country)
        steps_to_predict = len(actual_values)
        predictions_dict = autoregressive_predict(actual_values, inputs, steps_to_predict, dict)
        prediction = predictions_dict[variable]
        fitted_values = pd.DataFrame(prediction, columns=[variable])
        mse = hp.calculate_mse(actual_values, fitted_values, variable, 0)
        total_squared_error += mse * len(actual_values[variable]) 
        
        mae = hp.calculate_mae(actual_values, fitted_values, variable)
        total_absolute_error += mae * len(actual_values[variable])
        
        total_samples += len(actual_values[variable])
        
        # print(f"{country} MSE: {mae}")
    
    total_mse = total_squared_error / total_samples
    rmse = np.sqrt(total_mse)
    total_mae = total_absolute_error / total_samples
    return [total_mse, rmse, total_mae]