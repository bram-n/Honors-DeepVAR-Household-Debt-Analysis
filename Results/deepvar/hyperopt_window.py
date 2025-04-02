import numpy as np
import torch
import optuna
from functools import partial
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from torch.utils.data import DataLoader
import Results.deepvar.lstm as lstm
from Results.deepvar.lstm import LSTM_Model, TimeSeriesDataset

def objective(trial, X_train, y_train, X_val, y_val, input_size):
    """
    Objective function for Optuna optimization.
    
    Parameters:
    -----------
    trial : optuna.trial.Trial
        Optuna trial object
    X_train, y_train : torch.Tensor
        Training data
    X_val, y_val : torch.Tensor
        Validation data
    input_size : int
        Input size for the LSTM model
        
    Returns:
    --------
    float
        Validation loss
    """
    
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    hidden_size = trial.suggest_categorical("hidden_size", [16, 32, 64, 128])
    num_layers = trial.suggest_categorical("num_layers", [1, 2, 3])
    dropout = trial.suggest_categorical('dropout', [0.1, 0.2, 0.3, 0.4, 0.5])
    weight_decay = trial.suggest_categorical('weight_decay', [1e-5, 1e-4, 1e-3, 1e-2])
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])
    
    
    train_dataset = TimeSeriesDataset(X_train, y_train)
    val_dataset = TimeSeriesDataset(X_val, y_val)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=0,
        persistent_workers=False
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=0,
        persistent_workers=False
    )
    
    # Create model
    model = LSTM_Model(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=1,
        num_layers=num_layers,
        learning_rate=learning_rate,
        weight_decay=weight_decay
    )
    
    for layer in model.modules():
        if isinstance(layer, torch.nn.Dropout):
            layer.p = dropout
    
    # Early stopping callback
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        patience=5,
        verbose=False,
        mode='min'
    )
    
    trainer = pl.Trainer(
        max_epochs=500,
        callbacks=[early_stop_callback],
        enable_progress_bar=False,
        enable_model_summary=False,
        accelerator='cpu',
        devices=1,
        log_every_n_steps=10,
        deterministic=True
    )
    

    trainer.fit(model, train_loader, val_loader)
    
    val_loss = trainer.callback_metrics["val_loss"].item()
    
    return val_loss

def optimize_hyperparameters(df, inputs, output, n_trials=100):
    """
    Optimize hyperparameters for the LSTM model.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe
    inputs : list
        List of input column names
    output : str
        Output column name
    n_trials : int
        Number of optimization trials
        
    Returns:
    --------
    dict
        Best hyperparameters
    """
    lstm.set_seed()
    
    (X_train, y_train), (X_val, y_val), _, _, _ = lstm.prepare_lstm_data_with_windows(df, inputs, output)
    

    objective_func = partial(
        objective,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        input_size=X_train.shape[2]
    )
    
    study = optuna.create_study(direction="minimize")
    study.optimize(objective_func, n_trials=n_trials)
    

    print("Best trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
    
    return trial.params

def train_optimized_model(df, inputs, output, params):
    lstm.set_seed()
    
    learning_rate = params["learning_rate"]
    hidden_size = params["hidden_size"]
    num_layers = params["num_layers"]
    weight_decay = params["weight_decay"]
    
    model, predictions, errors, scaler_X, scaler_y = lstm.train_lstm_model_windows(
        df, 
        inputs, 
        output, 
        learning_rate=learning_rate,
        hidden_size=hidden_size,
        num_layers=num_layers,
        weight_decay=weight_decay,
        num_epochs=500,
        batch_size=params["batch_size"],
        verbose=True
    )
    
    return model, predictions, errors, scaler_X, scaler_y

def optimize_all_variables(df, variables, inputs, n_trials=50):
    best_params = {}
    
    for variable in variables:
        print(f"Optimizing hyperparameters for {variable}")
        params = optimize_hyperparameters(df, inputs, [variable], n_trials)
        best_params[variable] = params
    
    return best_params

def get_optimized_models(df, variables, inputs, best_params):
    models_dict = {}
    
    for variable in variables:
        print(f"Training model for {variable} with optimized hyperparameters...")
        model, predictions, errors, scaler_X, scaler_y = train_optimized_model(
            df, inputs, [variable], best_params[variable]
        )
        models_dict[variable] = [model, predictions, errors, scaler_X, scaler_y]
    
    return models_dict 