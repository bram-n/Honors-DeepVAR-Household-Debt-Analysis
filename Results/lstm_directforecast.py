import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import random
import pytorch_lightning as pl
import helpers as hp
import matplotlib.pyplot as plt


def get_lstm_input(df, lags):
    variables_unlagged = df.columns
    list = []
    for variable in variables_unlagged:
        for lag in range(1,lags + 1):
            list.append(f'{variable}_lag{lag}')
    return list


def create_lstm_data(df, lags):
    final_data = pd.DataFrame()
    countries = df.index.get_level_values('Country').unique()
    # For loop to prevent data leakage between countries
    for country in countries:
        country_data = hp.get_country(df, country).copy()
        for col in df.columns:
            for lag in range(1,lags+1):
                country_data[f'{col}_lag{lag}'] = country_data[col].shift(lag)
        final_data = pd.concat([final_data, country_data], axis=0)
        # final_data = final_data.dropna()
    return final_data


class LSTM_Model(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1): 
        super(LSTM_Model, self).__init__()
        set_seed()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, 
                           batch_first=True, dropout=0.5)
        
        self.batch_norm = nn.BatchNorm1d(hidden_size)
        self.dropout = nn.Dropout(0.5)  
        
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]
        out = self.batch_norm(out)
        out = self.dropout(out)
        out = self.fc(out)
        return out


def set_seed(seed=18):
    """Set all random seeds for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)
    pl.seed_everything(seed)

class TimeSeriesDataset(Dataset):
    """Dataset class for time series data."""
    def __init__(self, X: torch.Tensor, y: torch.Tensor):
        self.X = X
        self.y = y
        
    def __len__(self) -> int:
        return len(self.X)
    
    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]

def train_val_test_split(df, val_fraction=0.85):
    """
    Split the data into training, validation, and test sets while preserving time order
    """
    train, test = hp.time_panel_split_predict(df)
    train_final = pd.DataFrame()
    val = pd.DataFrame()
    countries = df.index.get_level_values('Country').unique()
    
    for country in countries:
        country_train_data = hp.get_country(train, country).sort_values(by='TIME_PERIOD')
        split_idx = int(len(country_train_data) * val_fraction)
        
        # Add this country's training data to the final training set
        train_final = pd.concat([train_final, country_train_data.iloc[:split_idx]])
        
        # Add this country's validation data to the validation set
        val = pd.concat([val, country_train_data.iloc[split_idx:]])
    
    train_final = train_final.dropna()
    val = val.dropna()
    test = test.dropna()
    
    return train_final, val, test


def prepare_direct_forecast_data(df, inputs, output, window_size, forecast_horizon):
    """
    Prepare time series data for direct multi-step forecasting.
    Instead of predicting t+1, we directly predict t+h where h is the forecast horizon.
    
    Args:
        df: DataFrame with multiindex (Country, Date)
        inputs: List of input column names
        output: Column name to predict
        window_size: Size of input window
        forecast_horizon: How many steps ahead to predict directly
        
    Returns:
        Training, validation, and test tensors, scalers
    """
    set_seed()
    
    # Initialize scalers
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    # Split data set
    train, val, test = train_val_test_split(df)
    
    # Extract features and target
    X_train_notscaled = train[inputs]
    X_val_notscaled = val[inputs]
    X_test_notscaled = test[inputs]
    y_train_notscaled = train[output]
    y_val_notscaled = val[output]
    y_test_notscaled = test[output]
    
    # Scale
    X_train = scaler_X.fit_transform(X_train_notscaled)
    X_val = scaler_X.transform(X_val_notscaled)
    X_test = scaler_X.transform(X_test_notscaled)
    y_train = scaler_y.fit_transform(y_train_notscaled.values.reshape(-1, 1))
    y_val = scaler_y.transform(y_val_notscaled.values.reshape(-1, 1))
    y_test = scaler_y.transform(y_test_notscaled.values.reshape(-1, 1))
    
    # Get country indices
    train_country_indices = train.index.get_level_values('Country')
    val_country_indices = val.index.get_level_values('Country')
    test_country_indices = test.index.get_level_values('Country')
    
    # Initialize lists for windowed data
    X_train_windowed, y_train_windowed = [], []
    X_val_windowed, y_val_windowed = [], []
    X_test_windowed, y_test_windowed = [], []
    
    # Get unique countries
    countries = train.index.get_level_values('Country').unique()
    
    # Process training data by country
    for country in countries:
        country_indices = np.where(train_country_indices == country)[0]
        X_train_country = X_train[country_indices]
        y_train_country = y_train[country_indices]
        
        if len(X_train_country) > window_size + forecast_horizon - 1:
            X_country_windowed, y_country_windowed = [], []
            for i in range(len(X_train_country) - window_size - forecast_horizon + 1):
                # Input window
                X_window = X_train_country[i:i+window_size]
                # Target is forecast_horizon steps ahead
                y_target = y_train_country[i+window_size+forecast_horizon-1]
                
                X_country_windowed.append(X_window)
                y_country_windowed.append(y_target)
                
            X_train_windowed.append(X_country_windowed)
            y_train_windowed.append(y_country_windowed)
    
    # Process validation data by country
    for country in countries:
        country_indices = np.where(val_country_indices == country)[0]
        X_val_country = X_val[country_indices]
        y_val_country = y_val[country_indices]
        
        if len(X_val_country) > window_size + forecast_horizon - 1:
            X_country_windowed, y_country_windowed = [], []
            for i in range(len(X_val_country) - window_size - forecast_horizon + 1):
                X_window = X_val_country[i:i+window_size]
                y_target = y_val_country[i+window_size+forecast_horizon-1]
                
                X_country_windowed.append(X_window)
                y_country_windowed.append(y_target)
                
            X_val_windowed.append(X_country_windowed)
            y_val_windowed.append(y_country_windowed)
    
    # Process test data by country
    for country in countries:
        country_indices = np.where(test_country_indices == country)[0]
        X_test_country = X_test[country_indices]
        y_test_country = y_test[country_indices]
        
        if len(X_test_country) > window_size + forecast_horizon - 1:
            X_country_windowed, y_country_windowed = [], []
            for i in range(len(X_test_country) - window_size - forecast_horizon + 1):
                X_window = X_test_country[i:i+window_size]
                y_target = y_test_country[i+window_size+forecast_horizon-1]
                
                X_country_windowed.append(X_window)
                y_country_windowed.append(y_target)
                
            X_test_windowed.append(X_country_windowed)
            y_test_windowed.append(y_country_windowed)
    
    # Combine data from all countries
    if X_train_windowed:
        X_train_windowed = np.concatenate(X_train_windowed, axis=0)
        y_train_windowed = np.concatenate(y_train_windowed, axis=0)
    else:
        raise ValueError("Not enough training data for the specified window size and forecast horizon")
        
    if X_val_windowed:
        X_val_windowed = np.concatenate(X_val_windowed, axis=0)
        y_val_windowed = np.concatenate(y_val_windowed, axis=0)
    else:
        raise ValueError("Not enough validation data for the specified window size and forecast horizon")
        
    if X_test_windowed:
        X_test_windowed = np.concatenate(X_test_windowed, axis=0)
        y_test_windowed = np.concatenate(y_test_windowed, axis=0)
    else:
        raise ValueError("Not enough test data for the specified window size and forecast horizon")
    
    # Convert to PyTorch tensors
    X_train = torch.from_numpy(X_train_windowed).float()
    y_train = torch.from_numpy(y_train_windowed).float()
    X_val = torch.from_numpy(X_val_windowed).float()
    y_val = torch.from_numpy(y_val_windowed).float()
    X_test = torch.from_numpy(X_test_windowed).float()
    y_test = torch.from_numpy(y_test_windowed).float()
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test), scaler_X, scaler_y

def train_direct_lstm_model(df, inputs, output_col, forecast_horizon, input_window_size=5,
                           learning_rate=0.0005, hidden_size=16, num_epochs=100, 
                           batch_size=64, verbose=True, predict_full_test=False):
    """
    Train a direct multi-step LSTM forecasting model.
    
    Args:
        df: DataFrame with multiindex (Country, Date)
        inputs: List of input column names
        output_col: Column name to predict
        forecast_horizon: How many steps ahead to predict directly 
        input_window_size: Size of input window
        learning_rate, hidden_size, etc.: Model hyperparameters
        predict_full_test: If True, ensure predictions are generated for the entire test period
        
    Returns:
        Trained model, predictions, errors, and scalers
    """
    set_seed()
    
    # Prepare data for direct forecasting
    (X_train, y_train), (X_val, y_val), (X_test, y_test), scaler_X, scaler_y = prepare_direct_forecast_data(
        df, inputs, output_col, input_window_size, forecast_horizon
    )
    
    # Input size is the number of features
    input_size = X_train.shape[2]
    
    # Create datasets and loaders
    train_dataset = TimeSeriesDataset(X_train, y_train)
    val_dataset = TimeSeriesDataset(X_val, y_val)
    test_dataset = TimeSeriesDataset(X_test, y_test)

    # Add drop_last=True to prevent batches of size 1 which can cause BatchNorm issues
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    # For test data we'll process one batch at a time to use all samples
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model - same LSTM architecture
    model = LSTM_Model(input_size=input_size, 
                      hidden_size=hidden_size, 
                      output_size=1) 

    # Optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.8, patience=3, verbose=True
    )
    
    criterion = nn.MSELoss()

    train_losses = []
    val_losses = []

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        epoch_train_loss = 0
        train_batch_count = 0
        
        for X, y in train_loader:
            # Skip batches that are too small
            if X.size(0) <= 1:
                continue
                
            optimizer.zero_grad()
            output = model(X)
            loss = criterion(output, y)
            loss.backward()
            
            # Add gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            epoch_train_loss += loss.item()
            train_batch_count += 1
        
        if train_batch_count == 0:
            print("Warning: No valid batches in training. Try reducing batch size.")
            continue
            
        avg_train_loss = epoch_train_loss / train_batch_count
        train_losses.append(avg_train_loss)

        # Validation phase
        model.eval()
        epoch_val_loss = 0
        val_batch_count = 0
        
        with torch.no_grad():
            for X, y in val_loader:
                # Skip batches that are too small
                if X.size(0) <= 1:
                    continue
                    
                output = model(X)
                val_loss = criterion(output, y)
                epoch_val_loss += val_loss.item()
                val_batch_count += 1
        
        if val_batch_count == 0:
            print("Warning: No valid batches in validation. Try reducing batch size.")
            continue
            
        avg_val_loss = epoch_val_loss / val_batch_count
        val_losses.append(avg_val_loss)

        # Update learning rate
        scheduler.step(avg_val_loss)
             
        if verbose and (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')
    
    # Plotting losses
    plot_losses(train_losses, val_losses, title=f"Direct Forecast Model (Horizon: {forecast_horizon})")
    
    # Evaluation of the model
    model.eval()
    all_predictions = []
    all_actuals = []
    
    with torch.no_grad():
        for X, y in test_loader:
            # For testing, we handle small batches carefully
            if X.size(0) <= 1:
                # If your model has BatchNorm layers, switch to eval mode to use running statistics
                model.eval()
            
            predictions = model(X)
            all_predictions.append(predictions)
            all_actuals.append(y)
        
        # Concatenate all batches
        final_predictions = torch.cat(all_predictions, dim=0)
        actual_values = torch.cat(all_actuals, dim=0)
        
        # Inverse transform to get actual values
        predictions_np = scaler_y.inverse_transform(final_predictions.numpy())
        actuals_np = scaler_y.inverse_transform(actual_values.numpy())
    
    mse = np.mean((predictions_np - actuals_np) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(predictions_np - actuals_np))
    
    print(f'Final LSTM Metrics (Horizon: {forecast_horizon}):')
    print('MSE:', mse)
    print('RMSE:', rmse)
    print('MAE:', mae)

    errors = [mse, rmse, mae]
    
    # Store the output column name for later use
    model.output_col = output_col
    model.forecast_horizon = forecast_horizon
    
    return model, predictions_np, errors, scaler_X, scaler_y, forecast_horizon

def direct_forecast_future(df, models_dict, output_col, input_window_size, future_steps):
    """
    Generate future predictions using multiple direct forecast models.
    
    Args:
        df: Historical dataframe containing the most recent data
        models_dict: Dictionary of models for different horizons {horizon: [model, ...]}
        output_col: Target column to predict
        input_window_size: Size of the input window
        future_steps: Number of steps to predict
        
    Returns:
        DataFrame with future predictions
    """
    set_seed()
    
    # Get the most recent data for each country
    countries = df.index.get_level_values('Country').unique()
    all_predictions = {}
    
    for country in countries:
        country_data = df.xs(country, level='Country')
        
        # Check if enough data is available
        if len(country_data) < input_window_size:
            print(f"Not enough data for {country}. Skipping.")
            continue
        
        # Get the most recent window of data
        recent_data = country_data.values[-input_window_size:]
        
        # Predictions for this country
        country_predictions = [None] * future_steps
        
        # For each horizon, use the appropriate model
        for horizon, (model, _, _, scaler_X, _, _) in models_dict.items():
            if horizon <= future_steps:
                model.eval()
                
                # Scale the input
                recent_data_scaled = scaler_X.transform(recent_data)
                input_tensor = torch.from_numpy(recent_data_scaled).float().unsqueeze(0)
                
                # Get prediction
                with torch.no_grad():
                    pred = model(input_tensor)
                    
                    # Create dummy array for inverse transform
                    dummy = np.zeros((1, scaler_X.n_features_in_))
                    dummy[0, model.target_idx] = pred.numpy().flatten()[0]
                    
                    # Get the actual prediction value
                    pred_value = scaler_X.inverse_transform(dummy)[0, model.target_idx]
                    
                    # Store in the right position (horizon-1 because horizons are 1-indexed)
                    country_predictions[horizon-1] = pred_value
        
        # Fill in any missing predictions
        if None in country_predictions:
            print(f"Warning: Some horizons missing for {country}. Using previous values.")
            # Forward fill predictions
            for i in range(future_steps):
                if country_predictions[i] is None:
                    # Use last available prediction or last observed value
                    if i > 0 and country_predictions[i-1] is not None:
                        country_predictions[i] = country_predictions[i-1]
                    else:
                        country_predictions[i] = country_data[output_col].iloc[-1]
        
        all_predictions[country] = country_predictions
    
    # Create DataFrame from predictions
    forecast_dfs = []
    
    for country, preds in all_predictions.items():
        # Get the last date in the original data
        last_date = df.xs(country, level='Country').index[-1]
        
        # Create new dates for forecasting
        import pandas as pd
        new_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=future_steps)
        
        # Create DataFrame for this country
        forecast_df = pd.DataFrame({
            'Country': [country] * future_steps,
            'Date': new_dates,
            output_col: preds
        })
        forecast_df.set_index(['Country', 'Date'], inplace=True)
        forecast_dfs.append(forecast_df)
    
    # Combine forecasts
    if forecast_dfs:
        return pd.concat(forecast_dfs)
    else:
        return pd.DataFrame()

def train_direct_multi_horizon_models(df, output_col, horizons, input_window_size=5, param_dict=None):
    """
    Train LSTM models for multiple direct forecast horizons
    
    Args:
        df: Input DataFrame
        output_col: Target variable to predict
        horizons: List of forecast horizons to train models for
        input_window_size: Size of the input window
        param_dict: Dictionary of hyperparameters {horizon: {'learning_rate': x, 'epochs': y}}
        
    Returns:
        Dictionary with models and scalers for each horizon
    """
    if param_dict is None:
        param_dict = {h: {'learning_rate': 0.0005, 'epochs': 100} for h in horizons}
    
    models_dict = {}
    
    for horizon in horizons:
        print(f"\nTraining model for horizon {horizon}")
        learning_rate = param_dict[horizon]['learning_rate']
        epochs = param_dict[horizon]['epochs']
        
        model, predictions, errors, scaler_X, scaler_y, _ = train_direct_lstm_model(
            df, 
            output_col=output_col,
            forecast_horizon=horizon,
            input_window_size=input_window_size,
            learning_rate=learning_rate,
            num_epochs=epochs
        )
        
        models_dict[horizon] = [model, predictions, errors, scaler_X, scaler_y, horizon]
    
    return models_dict


def plot_losses(train_losses, val_losses, title="Training and Validation Loss"):
    """Plot training and validation losses"""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss', color='blue')
    plt.plot(val_losses, label='Validation Loss', color='red')
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()