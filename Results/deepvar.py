import torch
import numpy as np
import pandas as pd
import lstm
import helpers as hp

def predict_next(model, scaler_X, scaler_y, current_input):
    scaled_input = scaler_X.transform(current_input)
    
    if torch.backends.mps.is_available() and next(model.parameters()).device.type == 'mps':
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
        
    input_tensor = torch.FloatTensor(scaled_input).unsqueeze(0).to(device)

    prediction = model(input_tensor)
    
    if prediction.device.type == 'mps':
        prediction_cpu = prediction.detach().cpu().numpy()
    else:
        prediction_cpu = prediction.detach().numpy()
    
    actual_prediction = scaler_y.inverse_transform(prediction_cpu.reshape(1, -1))[0][0]
    return actual_prediction
    
def autoregressive_predict(df, inputs, steps_to_predict, dict, LAG):
    lstm.set_seed()
    # Evaluate all models
    for model, _, _, _, _ in dict.values():
        model.eval()
    
    # Prepare initial input
    input_data = df.iloc[0][inputs].to_frame().T
    current_input = input_data.to_numpy().reshape(1, -1)
    
    # Initialize predictions dictionary
    predictions_dict = {key: [] for key in dict.keys()}
    
    with torch.no_grad(): #no grad since predicting
        for _ in range(steps_to_predict):
            new_row = {}
            for variable, (model, _, _, scaler_X, scaler_y) in dict.items():
                # Predict for each variable
                prediction = predict_next(model, scaler_X, scaler_y, current_input)
                
                # Store prediction
                predictions_dict[variable].append(prediction)
                
                # Prepare new row with lagged values
                new_row[f'{variable}_lag1'] = prediction
                for lag in range(2, LAG+1):
                    new_row[f'{variable}_lag{lag}'] = input_data[f'{variable}_lag{lag-1}'].values[0]
            # Update input data for next iteration
            input_data = pd.DataFrame([new_row])
            current_input = input_data.iloc[0].to_numpy().reshape(1, -1)
    
    return predictions_dict

def test_errors(df, test, variable, dictionary, inputs, LAG):
    lstm.set_seed()
    countries = df.index.get_level_values('Country').unique()
    df = df.drop(columns = inputs)
    total_squared_error = 0
    total_absolute_error = 0 
    total_samples = 0
    inputs = df.columns
    for country in countries:
        actual_values = hp.get_country(test, country)
        steps_to_predict = len(actual_values)
        predictions_dict = autoregressive_predict(actual_values, inputs, steps_to_predict, dictionary, LAG)
        prediction = predictions_dict[variable]
        fitted_values = pd.DataFrame(prediction, columns=[variable])
        mse = hp.calculate_mse(actual_values, fitted_values, variable)
        total_squared_error += mse * len(actual_values[variable]) 
        
        mae = hp.calculate_mae(actual_values, fitted_values, variable)
        total_absolute_error += mae * len(actual_values[variable])
        
        total_samples += len(actual_values[variable])
        
        # print(f"{country} MSE: {mae}")
    
    total_mse = total_squared_error / total_samples
    rmse = np.sqrt(total_mse)
    total_mae = total_absolute_error / total_samples
    return [total_mse, rmse, total_mae]


'''Gets the model, the X scaler and the y scaler in that order and stores it in a dictionary'''
def get_model_and_scaler_window(df, variables, inputs, param_dict, lag):
    dictionary = {}
    for variable in variables:
        learning_rate = param_dict[variable]['learning_rate']
        epochs = param_dict[variable]['epochs']
        model, predictions, errors, scaler_X, scaler_y = lstm.train_lstm_model_windows(
            df, inputs, [variable], learning_rate=learning_rate, num_epochs=epochs, windows=lag
        )        
        dictionary[variable] = [model, predictions, errors, scaler_X, scaler_y]
    return dictionary


def get_model_and_scaler_no_window(df, variables, inputs, param_dict):
    dictionary = {}
    for variable in variables:
        learning_rate = param_dict[variable]['learning_rate']
        epochs = param_dict[variable]['epochs']
        model, predictions, errors, scaler_X, scaler_y = lstm.train_lstm_model(df, inputs, [variable], learning_rate, num_epochs=epochs)
        dictionary[variable] = [model, predictions, errors, scaler_X, scaler_y]
    return dictionary