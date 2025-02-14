import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import helpers as hp
import matplotlib.pyplot as plt
import random
import pytorch_lightning as pl
# from laplace import Laplace

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


class LSTM_Model(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1): 
        super(LSTM_Model, self).__init__()
        torch.manual_seed(10)

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


def prepare_lstm_data(all_lstm_data, inputs, output):
    set_seed()
    
    # Initialize scalers
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    #Split data set
    lstm_train, lstm_test = train_test_split(all_lstm_data)

    X_train_notscaled = lstm_train[inputs]
    X_test_notscaled = lstm_test[inputs]

    y_train_notscaled = lstm_train[output]
    y_test_notscaled = lstm_test[output]

    # Scale parameters for more efficient optimization
    X_train = scaler_X.fit_transform(X_train_notscaled)
    X_test = scaler_X.transform(X_test_notscaled)
    y_train = scaler_y.fit_transform(y_train_notscaled .values.reshape(-1, 1))
    y_test = scaler_y.transform(y_test_notscaled.values.reshape(-1, 1))

    # Convert to pytorch tensor
    X_train = torch.from_numpy(X_train).float()
    y_train = torch.from_numpy(y_train).float()
    X_test = torch.from_numpy(X_test).float()
    y_test = torch.from_numpy(y_test).float()

    X_train = X_train.unsqueeze(1)
    X_test = X_test.unsqueeze(1)
    return X_train, X_test, y_train, y_test, scaler_X, scaler_y


def prepare_sliding_windows(X, y, window_size):
    set_seed()

    X_windowed = []
    y_windowed = []
    for i in range(len(X) - window_size + 1):
            X_window = X[i:i+window_size]
            y_window = y[i+window_size-1]
            X_windowed.append(X_window)
            y_windowed.append(y_window)
    return X_windowed, y_windowed

def prepare_lstm_data_with_windows(all_lstm_data, inputs, output, window_size):
    set_seed()

    # Initialize scalers
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    # Split data set
    lstm_train, lstm_test = train_test_split(all_lstm_data)
    X_train_notscaled = lstm_train[inputs]
    X_test_notscaled = lstm_test[inputs]
    y_train_notscaled = lstm_train[output]
    y_test_notscaled = lstm_test[output]
    
    # Scale parameters for more efficient optimization
    X_train_scaled = scaler_X.fit_transform(X_train_notscaled)
    X_test_scaled = scaler_X.transform(X_test_notscaled)
    y_train_scaled = scaler_y.fit_transform(y_train_notscaled.values.reshape(-1, 1))
    y_test_scaled = scaler_y.transform(y_test_notscaled.values.reshape(-1, 1))
    
    train_country_indices = lstm_train.index.get_level_values('Country')
    test_country_indices = lstm_test.index.get_level_values('Country')
    
    X_train_windowed, y_train_windowed = [], []
    countries = lstm_train.index.get_level_values('Country').unique()
    
    for country in countries:
        country_indices = np.where(train_country_indices == country)[0]
        X_train_country = X_train_scaled[country_indices]
        y_train_country = y_train_scaled[country_indices]
        X_country_windowed, y_country_windowed = prepare_sliding_windows(X_train_country, y_train_country, window_size)
        X_train_windowed.append(X_country_windowed)
        y_train_windowed.append(y_country_windowed)
    
    X_train_windowed = np.concatenate(X_train_windowed, axis=0)
    y_train_windowed = np.concatenate(y_train_windowed, axis=0)
    
    X_test_windowed, y_test_windowed = [], []
    for country in countries:
        country_indices = np.where(test_country_indices == country)[0]
        X_country = X_test_scaled[country_indices]
        y_country = y_test_scaled[country_indices]
        X_country_windowed, y_country_windowed = prepare_sliding_windows(X_country, y_country, window_size)
        X_test_windowed.append(X_country_windowed)
        y_test_windowed.append(y_country_windowed)
    
    X_test_windowed = np.concatenate(X_test_windowed, axis=0)
    y_test_windowed = np.concatenate(y_test_windowed, axis=0)
    
    #Convert to pytorch tensor
    X_train = torch.from_numpy(X_train_windowed).float()
    y_train = torch.from_numpy(y_train_windowed).float()
    X_test = torch.from_numpy(X_test_windowed).float()
    y_test = torch.from_numpy(y_test_windowed).float()
    
    return X_train, X_test, y_train, y_test, scaler_X, scaler_y


def train_lstm_model(all_lstm_data, inputs, output,
                     learning_rate=0.0005,  
                     hidden_size=16,  
                     output_size=1, 
                     num_epochs=100, 
                     batch_size=64,  
                     weight_decay=0.1,  
                     verbose=True):
    
    set_seed()

    X_train, X_test, y_train, y_test, scaler_X, scaler_y = prepare_lstm_data(all_lstm_data, inputs, output)

    train_dataset_lstm = TimeSeriesDataset(X_train, y_train)
    val_dataset_lstm = TimeSeriesDataset(X_test, y_test)

    train_loader_lstm = DataLoader(train_dataset_lstm, batch_size=batch_size, shuffle=True)
    val_loader_lstm = DataLoader(val_dataset_lstm, batch_size=batch_size, shuffle=False)
    
    model_lstm = LSTM_Model(input_size=X_train.shape[2], 
                           hidden_size=hidden_size, 
                           output_size=output_size)

    optimizer_lstm = torch.optim.Adam(model_lstm.parameters(), lr = learning_rate)
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_lstm, 
                                                         mode='min', 
                                                         factor=0.8, 
                                                         patience=3, 
                                                         verbose=True)
    
    criterion_lstm = nn.MSELoss()

    train_losses_lstm = []
    val_losses_lstm = []

    for epoch in range(num_epochs):
        # Training phase with gradient clipping
        model_lstm.train()
        epoch_train_loss = 0
        train_batch_count = 0
        
        for X, y in train_loader_lstm:
            optimizer_lstm.zero_grad()
            output = model_lstm(X)
            loss = criterion_lstm(output, y)
            loss.backward()
            
            # Add gradient clipping
            torch.nn.utils.clip_grad_norm_(model_lstm.parameters(), max_norm=1.0)
            
            optimizer_lstm.step()
            epoch_train_loss += loss.item()
            train_batch_count += 1
        
        avg_train_loss = epoch_train_loss / train_batch_count
        train_losses_lstm.append(avg_train_loss)

        # Validation phase
        model_lstm.eval()
        epoch_val_loss = 0
        val_batch_count = 0
        
        with torch.no_grad():
            for X, y in val_loader_lstm:
                output = model_lstm(X)
                val_loss = criterion_lstm(output, y)
                epoch_val_loss += val_loss.item()
                val_batch_count += 1
        
        avg_val_loss = epoch_val_loss / val_batch_count
        val_losses_lstm.append(avg_val_loss)

        # Update learning rate
        scheduler.step(avg_val_loss)
             
        if verbose and (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')
            

    plot_losses(train_losses_lstm, val_losses_lstm)
    
    # Evaluation of the model
    model_lstm.eval()
    with torch.no_grad():
        final_predictions = model_lstm(X_test)
        predictions_lstm = scaler_y.inverse_transform(final_predictions.numpy())
        actuals_lstm = scaler_y.inverse_transform(y_test.numpy())
    
    mse_lstm = np.mean((predictions_lstm - actuals_lstm) ** 2)
    rmse_lstm = np.sqrt(mse_lstm)
    mae_lstm = np.mean(np.abs(predictions_lstm - actuals_lstm))
    
    print('Final LSTM Metrics:')
    print('MSE:', mse_lstm)
    print('RMSE:', rmse_lstm)
    print('MAE:', mae_lstm)

    errors = [mse_lstm, rmse_lstm, mae_lstm]

    return model_lstm, predictions_lstm, errors, scaler_X, scaler_y

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
    for country in countries:
        country_data = hp.get_country(df, country).copy()
        for col in df.columns:
            for lag in range(1,lags+1):
                country_data[f'{col}_lag{lag}'] = country_data[col].shift(lag)
        final_data = pd.concat([final_data, country_data], axis=0)
        # final_data = final_data.dropna()
    return final_data


def train_test_split(df):
    train, test = hp.time_panel_split_predict(df)
    train = train.dropna()
    test = test.dropna()
    return train, test

def train_val_test_split(df, val_ratio=0.15):
    """
    Split the data into training, validation, and test sets while preserving time order
    """
    train, test = hp.time_panel_split_predict(df)
    
    # Calculate split point for validation set
    train_len = len(train)
    val_size = int(train_len * val_ratio)
    
    # Split training data into train and validation
    val_split_idx = train_len - val_size
    
    # Split while maintaining index
    train_final = train.iloc[:val_split_idx]
    val = train.iloc[val_split_idx:]
    
    # Drop any rows with NaN values
    train_final = train_final.dropna()
    val = val.dropna()
    test = test.dropna()
    
    return train_final, val, test

def get_LSTM_RMSE_TOTAL(model, all_lstm_data, inputs, output):
    
    scaler_X_lstm = StandardScaler()
    scaler_y_lstm = StandardScaler()

    X_full_notscaled = all_lstm_data[inputs]
    y_full_notscaled = all_lstm_data[output]

    X_full_lstm = scaler_X_lstm.fit_transform(X_full_notscaled)
    y_full_lstm = scaler_y_lstm.fit_transform(y_full_notscaled.values.reshape(-1, 1))
    
    X_full_lstm = torch.from_numpy(X_full_lstm).float().unsqueeze(1)
    y_full_lstm = torch.from_numpy(y_full_lstm).float()
    
    model.eval()
    with torch.no_grad():
        full_predictions_lstm = model(X_full_lstm)
        full_predictions = scaler_y_lstm.inverse_transform(full_predictions_lstm.numpy())
        full_actuals = scaler_y_lstm.inverse_transform(y_full_lstm.numpy())
        
        full_mse = np.mean((full_predictions - full_actuals) ** 2)
        full_rmse = np.sqrt(full_mse)
        full_mae = np.mean(np.abs(full_predictions - full_actuals))
        
        # print('LSTM Metrics for Full Dataset:')
        # print('MSE:', full_mse)
        # print('RMSE:', full_rmse)
        # print('MAE:', full_mae)
        return full_rmse
    


def fill_forecast_values(df, inputs, variable, dict):
    model, _, _, scaler_X, scaler_y = dict[variable]
    forecast_features = df[inputs]    
        
    X_scaled = scaler_X.transform(forecast_features)

    X_tensor = torch.from_numpy(X_scaled).float()
    X_tensor = X_tensor.unsqueeze(1)  

    model.eval()
    with torch.no_grad():
        scaled_predictions = model(X_tensor)
        
    predictions = scaler_y.inverse_transform(scaled_predictions.numpy())

    final_predictions = pd.DataFrame(predictions.flatten(), columns=[variable])
    final_predictions = final_predictions.set_index(df.index)

    return final_predictions


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
