import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
import Results.tools.helpers as hp
import matplotlib.pyplot as plt
import random
from pytorch_lightning.tuner import Tuner
import pytorch_lightning as pl
import torch
import torch.nn as nn

def set_seed(seed=18, silent = True):
    """Set all random seeds for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)
    pl.seed_everything(seed, verbose=False) 

    
class TimeSeriesDataset(Dataset):
    """Dataset class for time series data."""
    def __init__(self, X: torch.Tensor, y: torch.Tensor):
        self.X = X
        self.y = y
        
    def __len__(self) -> int:
        return len(self.X)
    
    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


class LSTM_Model(pl.LightningModule):
    def __init__(self, 
                 input_size, 
                 hidden_size, 
                 output_size, 
                 num_layers=2, 
                 dropout = .1,
                 learning_rate=0.0005, 
                 weight_decay=0.1):
        super().__init__()
        
        self.save_hyperparameters()
        
        self.lstm = nn.LSTM(
            input_size=input_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            batch_first=True, 
            dropout=dropout
        )
        
        self.batch_norm = nn.BatchNorm1d(hidden_size)
        self.dropout = nn.Dropout(dropout)
        
        self.fc = nn.Linear(hidden_size, output_size)
        
        self.criterion = nn.MSELoss()

    def forward(self, x):
        h0 = torch.zeros(self.hparams.num_layers, x.size(0), self.hparams.hidden_size).to(x.device)
        c0 = torch.zeros(self.hparams.num_layers, x.size(0), self.hparams.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]  # Take the last time step
        out = self.batch_norm(out)
        out = self.dropout(out)
        out = self.fc(out)
        return out

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        val_loss = self.criterion(y_hat, y)
        self.log('val_loss', val_loss, prog_bar=True)
        return val_loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), 
            lr=self.hparams.learning_rate, 
            weight_decay=self.hparams.weight_decay
        )
        
        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            factor=0.8, 
            patience=3, 
            verbose=False,
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss'
            }
        }

def prepare_lstm_data(df, inputs, output, train_test_only = True):
    set_seed()
    
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    train, val, test = train_val_test_split(df)

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

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

    # Convert to tensors
    X_train = torch.from_numpy(X_train).float().unsqueeze(1)
    y_train = torch.from_numpy(y_train).float()
    
    X_val = torch.from_numpy(X_val).float().unsqueeze(1)
    y_val = torch.from_numpy(y_val).float()
    
    X_test = torch.from_numpy(X_test).float().unsqueeze(1)
    y_test = torch.from_numpy(y_test).float()

    return (X_train, y_train), (X_val, y_val), (X_test, y_test), scaler_X, scaler_y



def train_lstm_model(all_lstm_data, inputs, output,
 learning_rate=0.0005,
 hidden_size=16,
 output_size=1,
 num_epochs=100,
 batch_size=64,
 weight_decay=0.1,
 num_layers=1,
 dropout = .3,
 verbose=False):
    """
    Train LSTM model
    """
    device = torch.device("cpu")
    
    set_seed()
    # Prepare data
    (X_train, y_train), (X_val, y_val), (X_test, y_test), scaler_X, scaler_y = prepare_lstm_data(
        all_lstm_data, inputs, output
    )
    
    
    X_train = X_train.clone().detach().to(device)
    y_train = y_train.clone().detach().to(device)
    X_val = X_val.clone().detach().to(device)
    y_val = y_val.clone().detach().to(device)
    X_test = X_test.clone().detach().to(device)
    y_test = y_test.clone().detach().to(device)
    
    # Create datasets and dataloaders
    train_dataset = TimeSeriesDataset(X_train, y_train)
    val_dataset = TimeSeriesDataset(X_val, y_val)
    test_dataset = TimeSeriesDataset(X_test, y_test)
    
    
    train_loader = DataLoader(train_dataset, 
                            batch_size=batch_size, 
                            shuffle=True, 
                            num_workers=0,
                            persistent_workers=False)
    val_loader = DataLoader(val_dataset, 
                          batch_size=batch_size, 
                          shuffle=False, 
                          num_workers=0,
                          persistent_workers=False)
    test_loader = DataLoader(test_dataset, 
                           batch_size=batch_size, 
                           shuffle=False, 
                           num_workers=0,
                           persistent_workers=False)
    
    
    model = LSTM_Model(
        input_size=X_train.shape[2],
        hidden_size=hidden_size,
        output_size=output_size,
        num_layers=num_layers,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        dropout=dropout,
    ).to(device)
    
    # Early stopping callback
    early_stop_callback = pl.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        verbose=verbose,
        mode='min'
    )
    
    
    trainer = pl.Trainer(
        max_epochs=num_epochs,
        callbacks=[early_stop_callback],
        enable_progress_bar=False,
        enable_model_summary=False,
        accelerator='cpu',  
        devices=1,
        log_every_n_steps=10,
        deterministic=True
    )
    
    # Fit the model
    trainer.fit(model, train_loader, val_loader)
    
    # Use the trained model directly
    model.eval()
    
    # Predictions
    with torch.no_grad():
        test_predictions = model(X_test)
        
        
        test_predictions = test_predictions.cpu()
        predictions_lstm = scaler_y.inverse_transform(test_predictions.numpy())
        actuals_lstm = scaler_y.inverse_transform(y_test.cpu().numpy())
    
    # Metrics
    mse_lstm = np.mean((predictions_lstm - actuals_lstm) ** 2)
    rmse_lstm = np.sqrt(mse_lstm)
    mae_lstm = np.mean(np.abs(predictions_lstm - actuals_lstm))
    
    if verbose:
        print('Final LSTM Metrics:')
        print('MSE:', mse_lstm)
        print('RMSE:', rmse_lstm)
        print('MAE:', mae_lstm)
    
    errors = [mse_lstm, rmse_lstm, mae_lstm]
    return model, predictions_lstm, errors, scaler_X, scaler_y



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
    train, val, test = train_val_test_split(all_lstm_data)

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

    train_country_indices = train.index.get_level_values('Country')
    val_country_indices = val.index.get_level_values('Country')
    test_country_indices = test.index.get_level_values('Country')
    
    X_train_windowed, y_train_windowed = [], []
    X_val_windowed, y_val_windowed = [], []
    X_test_windowed, y_test_windowed = [], []

    countries = train.index.get_level_values('Country').unique()
    
    for country in countries:
        country_indices = np.where(train_country_indices == country)[0]
        X_train_country = X_train[country_indices]
        y_train_country = y_train[country_indices]
        X_country_windowed, y_country_windowed = prepare_sliding_windows(X_train_country, y_train_country, window_size)
        X_train_windowed.append(X_country_windowed)
        y_train_windowed.append(y_country_windowed)
    
    X_train_windowed = np.concatenate(X_train_windowed, axis=0)
    y_train_windowed = np.concatenate(y_train_windowed, axis=0)
    
    for country in countries:
        country_indices = np.where(val_country_indices == country)[0]
        X_country = X_val[country_indices]
        y_country = y_val[country_indices]
        X_country_windowed, y_country_windowed = prepare_sliding_windows(X_country, y_country, window_size)
        X_val_windowed.append(X_country_windowed)
        y_val_windowed.append(y_country_windowed)

    X_val_windowed = np.concatenate(X_val_windowed, axis=0)
    y_val_windowed = np.concatenate(y_val_windowed, axis=0)
    
    # for loop to prevent data leakage between the countries while still creating windows.
    for country in countries:
        country_indices = np.where(test_country_indices == country)[0]
        X_country = X_test[country_indices]
        y_country = y_test[country_indices]
        X_country_windowed, y_country_windowed = prepare_sliding_windows(X_country, y_country, window_size)
        X_test_windowed.append(X_country_windowed)
        y_test_windowed.append(y_country_windowed)
    
    X_test_windowed = np.concatenate(X_test_windowed, axis=0)
    y_test_windowed = np.concatenate(y_test_windowed, axis=0)
    
    #Convert to pytorch tensor
    X_train = torch.from_numpy(X_train_windowed).float()
    y_train = torch.from_numpy(y_train_windowed).float()
    X_val = torch.from_numpy(X_val_windowed).float()
    y_val = torch.from_numpy(y_val_windowed).float()
    X_test = torch.from_numpy(X_test_windowed).float()
    y_test = torch.from_numpy(y_test_windowed).float()
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test), scaler_X, scaler_y


def train_lstm_model_windows(all_lstm_data, inputs, output, 
                     windows = 3,
                     learning_rate=0.0005,  
                     hidden_size=16,  
                     output_size=1, 
                     num_epochs=100, 
                     batch_size=64,  
                     weight_decay=0.1,
                     dropout = .1,
                     num_layers = 1,
                     verbose=False):
    
    set_seed()

    # Force CPU usage
    device = torch.device("cpu")

    (X_train, y_train), (X_val, y_val), (X_test, y_test), scaler_X, scaler_y = prepare_lstm_data_with_windows(all_lstm_data, inputs, output, windows)

    X_train = X_train.clone().detach().to(device)
    y_train = y_train.clone().detach().to(device)
    X_val = X_val.clone().detach().to(device)
    y_val = y_val.clone().detach().to(device)
    X_test = X_test.clone().detach().to(device)
    y_test = y_test.clone().detach().to(device)
    
    # Create datasets and dataloaders
    train_dataset = TimeSeriesDataset(X_train, y_train)
    val_dataset = TimeSeriesDataset(X_val, y_val)
    test_dataset = TimeSeriesDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                            num_workers=0, persistent_workers=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                          num_workers=0, persistent_workers=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, 
                           num_workers=0, persistent_workers=False)
    
    model = LSTM_Model(
        input_size=X_train.shape[2],
        hidden_size=hidden_size,
        output_size=output_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        dropout = dropout,
        num_layers= num_layers
    ).to(device)
    
    early_stop_callback = pl.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        verbose=verbose,
        mode='min'
    )
    
    # Configure trainer to use CPU
    trainer = pl.Trainer(
        max_epochs=num_epochs,
        callbacks=[early_stop_callback],
        enable_progress_bar=False,
        enable_model_summary=False,
        accelerator='cpu',  # Force CPU
        devices=1,
        deterministic=True
    )
    
    # Fit the model
    trainer.fit(model, train_loader, val_loader)
    
    # Use the trained model directly
    model.eval()
    
    with torch.no_grad():
        test_predictions = model(X_test)
        
        # Already on CPU, but keep for clarity
        test_predictions = test_predictions.cpu()
        predictions_lstm = scaler_y.inverse_transform(test_predictions.numpy())
        actuals_lstm = scaler_y.inverse_transform(y_test.cpu().numpy())
    
    # Metrics
    mse_lstm = np.mean((predictions_lstm - actuals_lstm) ** 2)
    rmse_lstm = np.sqrt(mse_lstm)
    mae_lstm = np.mean(np.abs(predictions_lstm - actuals_lstm))
    
    if verbose:
        print('Final LSTM Metrics:')
        print('MSE:', mse_lstm)
        print('RMSE:', rmse_lstm)
        print('MAE:', mae_lstm)
    
    errors = [mse_lstm, rmse_lstm, mae_lstm]
    return model, predictions_lstm, errors, scaler_X, scaler_y

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
    # for loop to prevent data leakage between countries
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
        
        train_final = pd.concat([train_final, country_train_data.iloc[:split_idx]])
        
        val = pd.concat([val, country_train_data.iloc[split_idx:]])
    
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

    model = model.cpu()
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


def find_learning_rate(all_lstm_data, inputs, output,
                       hidden_size=16,
                       output_size=1,
                       batch_size=64,
                       min_lr=1e-5,
                       max_lr=1,
                       num_iter=100,
                       plot_results=True):
    """
    Find the optimal learning rate for the LSTM model using PyTorch Lightning's learning rate finder.
    """
    
    set_seed()
    
    (X_train, y_train), (X_val, y_val), (X_test, y_test), scaler_X, scaler_y = prepare_lstm_data(all_lstm_data, inputs, output)
    train_dataset = TimeSeriesDataset(X_train, y_train)

    # Force CPU usage
    device = torch.device("cpu")
    
    train_loader = DataLoader(train_dataset, 
                            batch_size=batch_size, 
                            shuffle=True, 
                            num_workers=0,
                            persistent_workers=False)
    
    class LightningLSTM(pl.LightningModule):
        def __init__(self, input_size, hidden_size, output_size, learning_rate=1e-3):
            super().__init__()
            self.save_hyperparameters()  
            self.model = LSTM_Model(input_size, hidden_size, output_size)
            self.criterion = nn.MSELoss()
          
        def forward(self, x):
            return self.model(x)
        
        def training_step(self, batch, batch_idx):
            x, y = batch
            y_hat = self(x)
            loss = self.criterion(y_hat, y)
            self.log('train_loss', loss)
            return loss
            
        def configure_optimizers(self):
            optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
            return optimizer
    
    # Create model with explicit learning rate
    model = LightningLSTM(
        input_size=X_train.shape[2], 
        hidden_size=hidden_size, 
        output_size=output_size,
        learning_rate=1e-3  
    )
    
    # Initialize trainer with CPU
    trainer = pl.Trainer(
        max_epochs=1,
        enable_progress_bar=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        accelerator='cpu',  # Force CPU
        devices=1
    )
    
    # Create a tuner for the trainer
    tuner = Tuner(trainer)
    
    # Run the learning rate finder with explicit range
    print("Finding optimal learning rate...")
    lr_finder = tuner.lr_find(
        model,
        train_dataloaders=train_loader,
        min_lr=min_lr,
        max_lr=max_lr,
        num_training=num_iter
    )
    
    suggested_lr = lr_finder.suggestion()
    
    if plot_results and lr_finder is not None:
        fig, ax = plt.subplots(figsize=(10, 6))
        lr_finder.plot(ax=ax, suggest=True)
        plt.xscale('log')
        plt.xlabel('Learning Rate (log scale)')
        plt.ylabel('Loss')
        plt.title('Learning Rate Finder Results')
        plt.tight_layout()
        plt.show()
    
    print(f"Suggested learning rate: {suggested_lr:.8f}")
    return suggested_lr