
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import Results.deepvar.lstm as lstm
import warnings 

import copy
import warnings

import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger
import numpy as np
import pandas as pd
import torch

from pytorch_forecasting import Baseline, TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import MAE, QuantileLoss, RMSE
from pytorch_forecasting.models.temporal_fusion_transformer.tuning import (
    optimize_hyperparameters,
)

from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data.encoders import GroupNormalizer

from lightning.pytorch.tuner import Tuner

import tensorflow as tf

def prepare_panel_data(df):
    df = df.copy()
    
    time_periods = pd.to_datetime(df.index.get_level_values('TIME_PERIOD'))
    
    df = df.reset_index().sort_values(['TIME_PERIOD', 'Country'])
    
    df['time_idx'] = df.groupby('Country').cumcount()
    
    df = df.set_index(['TIME_PERIOD', 'Country'])
    df["Country_index"] = df.index.get_level_values("Country")
    return df


def train_temporal_fusion_transformer(
    train_data,
    max_prediction_length,
    time_idx_col="time_idx",
    target_col="l_GDP_dif",
    group_ids=["Country_index"],
    max_encoder_length=3,
    known_reals=["time_idx", "l_GDP_dif_lag1", "hd_dif_lag1", "pd_dif_lag1"],
    batch_size=32,
    max_epochs=50,
    patience=30,
    learning_rate=None,  # If None, will use learning rate finder
    hidden_size=64,
    attention_head_size=4,
    dropout=0.2,
    hidden_continuous_size=32,
    limit_train_batches=30,
    device=None,
    loss="quantile", 
    seed = 42
):
    """
    Create and train a Temporal Fusion Transformer model for time series forecasting.
    
    Args:
        train_data: pandas DataFrame containing the training data
        time_idx_col: column name for the time index
        target_col: column name for the target variable
        group_ids: list of column names for grouping the time series
        max_encoder_length: maximum encoder length for the model
        max_prediction_length: maximum prediction length for the model
        known_reals: list of known real-valued features
        batch_size: batch size for training
        max_epochs: maximum number of epochs for training
        patience: patience for early stopping
        learning_rate: learning rate for optimizer (if None, will use learning rate finder)
        hidden_size: hidden size for the model
        attention_head_size: attention head size for the model
        dropout: dropout rate
        hidden_continuous_size: hidden continuous size for the model
        limit_train_batches: number of batches to use for training
        device: device to use for training (None for auto-detection)
        
    Returns:
        Trained TemporalFusionTransformer model and training metrics
    """

    if device is None:
        device = "gpu" if torch.cuda.is_available() else "cpu"
    
    pl.seed_everything(seed)
    
    training = TimeSeriesDataSet(
        train_data,
        time_idx=time_idx_col,
        target=target_col,
        group_ids=group_ids,
        min_encoder_length=max_encoder_length,
        max_encoder_length=max_encoder_length,
        max_prediction_length=max_prediction_length,
        static_categoricals=group_ids,
        static_reals=[],
        time_varying_known_categoricals=[],
        time_varying_known_reals=known_reals,
        time_varying_unknown_reals=[target_col],
        target_normalizer=GroupNormalizer(
            groups=group_ids,
            transformation=None
        ),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
    )
    
    validation = TimeSeriesDataSet.from_dataset(
        training, train_data, predict=True, stop_randomization=True
    )
    
    train_dataloader = training.to_dataloader(
        train=True, batch_size=batch_size, num_workers=0
    )
    val_dataloader = validation.to_dataloader(
        train=False, batch_size=batch_size * 10, num_workers=0
    )
    
    early_stop_callback = EarlyStopping(
        monitor="train_loss", min_delta=1e-2, patience=patience, verbose=False, mode="min"
    )
    lr_logger = LearningRateMonitor()
    logger = TensorBoardLogger("lightning_logs")
    
    if loss == "quantile":
        quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]
        loss_function = QuantileLoss(quantiles=quantiles)
    elif loss == "RMSE":
        loss_function = RMSE()
    elif loss == "MAE":
        loss_function = MAE()
    else:
        raise ValueError(f"Loss function {loss} not supported")

    
    model = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=0.03 if learning_rate is None else learning_rate,
        hidden_size=hidden_size,
        attention_head_size=attention_head_size,
        dropout=dropout,
        hidden_continuous_size=hidden_continuous_size,
        loss=loss_function,
        optimizer="Ranger",
        log_interval=0,
        reduce_on_plateau_patience=4,
    )
    
    model.to(device)
    print(f"Number of parameters in network: {model.size()/1e3:.1f}k")
    
    # Find optimal learning rate if not provided
    if learning_rate is None:
        lr_trainer = pl.Trainer(
            max_epochs=5,
            accelerator=device,
            devices=1,
            enable_progress_bar=False,
            enable_model_summary=False,
            logger=False
        )
        
        tuner = Tuner(lr_trainer)
        
        print("Finding optimal learning rate...")
        lr_finder = tuner.lr_find(
            model,
            train_dataloaders=train_dataloader,
            min_lr=1e-5,
            max_lr=0.1,
            num_training=100
        )
        
        fig = lr_finder.plot(suggest=True)
        suggested_lr = lr_finder.suggestion()
        print(f"Suggested learning rate: {suggested_lr}")
        
        model.hparams.learning_rate = suggested_lr
    
 
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator=device,
        devices=1,
        callbacks=[early_stop_callback, lr_logger],
        gradient_clip_val=0.1,
        limit_train_batches=limit_train_batches,
        logger=logger
    )
    
    trainer.fit(
        model,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader
    )
    
    return model, training


def get_predictions(test_dataloader, tft_model, DEVICE, loss="quantile"):
    """
    Get predictions from a trained Temporal Fusion Transformer model.
    
    Args:
        test_dataloader: DataLoader containing test data
        tft_model: Trained TFT model
        DEVICE: Device to run predictions on (cpu or cuda)
        loss: Loss type used for training ("quantile" or "rmse")
        
    Returns:
        tuple: (actuals, predictions, raw_predictions, x) where:
            - actuals: Tensor of actual values
            - predictions: Model predictions in requested mode
            - raw_predictions: Raw model outputs
            - x: Input features used for predictions
    """
    actuals = torch.cat([y[0].to(DEVICE) for x, y in iter(test_dataloader)])
    
    # Get prediction based on loss function input
    if loss == "quantile":
        predictions = tft_model.predict(
            test_dataloader, 
            mode="quantiles",
            trainer_kwargs=dict(accelerator="cpu"),
        )
    else:
        predictions = tft_model.predict(
            test_dataloader, 
            mode="prediction",
            trainer_kwargs=dict(accelerator="cpu"),
        )
    
    predictions = predictions.to(DEVICE)
    
    raw_predictions, x = tft_model.predict(
        test_dataloader, 
        mode="raw", 
        return_x=True,
        trainer_kwargs=dict(accelerator="cpu"),
    )[:2]
    
    return actuals, predictions, raw_predictions, x

def plot_country_predictions(country_name, raw_predictions, actuals, prediction_index, test_data, loss, target_var):
    """
    Plot predictions with quantiles for a specific country with improved styling
    
    Args:
        country_name: Name of the country to plot
        raw_predictions: Raw predictions tensor [39, 10, 5]
        test_data: Test data DataFrame
        country_idx: Index of the country in raw_predictions (if known)
    """
    predictions = get_country_tft_predicitons(test_data, raw_predictions, country_name)
        
    observed = get_country_tft_actual(test_data, actuals, country_name)
    dates = prediction_index
    print(prediction_index)
    if loss == "quantile":
        quantile_values = [0.1, 0.25, 0.5, 0.75, 0.9] 
        
        
        plt.figure(figsize=(14, 8))
        
        plt.style.use('seaborn-v0_8-whitegrid')
        
        plt.plot(dates[-len(predictions):], observed, label="Observed", color="#1F77B4", linewidth=2.5, marker='o', markersize=5)

        median_idx = 2  
        plt.plot(dates[-len(predictions):], predictions[:, median_idx], 'o-',
                label=f"Median Prediction (q={quantile_values[median_idx]})",
                color="#D62728", linewidth=2)
        
        quantile_colors = ["#AAAAAA", "#888888", "#D62728", "#888888", "#AAAAAA"]
        for i, q in enumerate(quantile_values):
            if i != median_idx:  
                plt.plot(dates[-len(predictions):], predictions[:, i], '--',
                        label=f"Quantile {q}", alpha=0.7, color=quantile_colors[i])
        
        plt.fill_between(
            dates[-len(predictions):],
            predictions[:, 0],  # 10th percentile
            predictions[:, -1],  # 90th percentile
            color="orange",
            alpha=0.3,
            label="80% Prediction Interval"
        )
        
        plt.xlabel("Time Period", fontsize=12, fontweight='bold')
        plt.ylabel(f"{target_var}", fontsize=12, fontweight='bold')
        plt.title(f"{country_name}: Observed vs. Predicted {target_var}", 
                fontsize=16, fontweight='bold', pad=20)
        

        plt.legend(loc='best', frameon=True, framealpha=0.9, fontsize=10)

        
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        plt.tick_params(axis='both', which='major', labelsize=10)
        
        plt.box(True)
        
        plt.tight_layout()
        
        plt.show()
    else:
        plt.figure(figsize=(14, 8))
        print("Observe", observed)
        print("Predicted", predictions)
        plt.plot(dates, observed, label="Observed", color="#1F77B4", linewidth=2.5, marker='o', markersize=5)
        plt.plot(dates, predictions, label="Predicted", color="#D62728", linewidth=2)
        plt.xlabel("Time Period", fontsize=12, fontweight='bold')
        plt.ylabel(f"{target_var}", fontsize=12, fontweight='bold')
        plt.title(f"{country_name}: Observed vs. Predicted {target_var}", fontsize=16, fontweight='bold', pad=20)
   


def predict_over_time_steps(df, tft_onestep, training, start_time_idx, num_steps, 
                            max_prediction_length_onestep=1, max_encoder_length_onestep=3,
                            batch_size=128, device=None):
    """
    Generate predictions over multiple time steps for all countries.
    
    Args:
        df (pd.DataFrame): The dataframe containing time series data
        tft_onestep (TemporalFusionTransformer): The trained model
        training (TimeSeriesDataSet): The training dataset used to create test datasets
        start_time_idx (int): Starting time index for prediction
        num_steps (int): Number of time steps to predict
        max_prediction_length_onestep (int): Prediction length for each step
        max_encoder_length_onestep (int): Encoder length for context
        batch_size (int): Batch size for dataloader
        device (str): Device to run predictions on ("cuda" or "cpu")
        
    Returns:
        tuple: Tuple containing (predictions tensor of shape [num_countries, num_steps, ...], 
                                list of actual values)
    """
    if device is None:
        device = "gpu" if torch.cuda.is_available() else "cpu"
    
    all_predictions_list = []
    all_actuals_list = []
    

    for step in range(num_steps):
        last_time_for_prediction_onestep = start_time_idx + step

        # Create test data 
        test = df[(df['time_idx'] >= (last_time_for_prediction_onestep - max_encoder_length_onestep)) & 
                  (df['time_idx'] <= last_time_for_prediction_onestep)]
        testing = TimeSeriesDataSet.from_dataset(
            training, test, predict=True, stop_randomization=True
        )
        
        test_dataloader = testing.to_dataloader(
            train=False, batch_size=batch_size * 10, num_workers=0
        )

        # gets the actual values for the test data
        all_actuals = torch.cat([y[0].to(device) for x, y in iter(test_dataloader)])
        # gets the raw predictions for the test data
        raw_predictions, x = tft_onestep.predict(
            test_dataloader, 
            mode="raw", 
            return_x=True,
            trainer_kwargs=dict(accelerator="cpu")
        )[:2]
       
        step_prediction = raw_predictions.prediction
        all_predictions_list.append(step_prediction)
        all_actuals_list.append(all_actuals)
        
        # print("SHAPE_Allpredictions", all_predictions_list[0].shape)
        # print(f"Completed prediction for time index {last_time_for_prediction_onestep}")

    all_predictions = torch.stack(all_predictions_list, dim=1)
    all_predictions = all_predictions.squeeze(-1) 

    return all_predictions, all_actuals_list

def plot_onestep_country_predictions(predictions, actuals, start_time_idx, country_name, target_var, loss, df):
    """
    Plot predictions with quantiles for a specific country with improved styling
    
    Args:
        predictions: Predictions tensor with quantiles
        actuals: List of actual values
        start_time_idx: Starting time index for predictions
    """
    country_index = get_country_index(df, country_name)

    country_predictions = predictions[country_index]
    country_predictions_length = len(country_predictions)
    indices = np.arange(start_time_idx, start_time_idx + country_predictions_length)

    date_level = 0  
    all_dates = df.index.get_level_values(date_level)

    dates = [all_dates[idx] for idx in indices]
    
    country_index = get_country_index(df, country_name)
    
    country_actuals = torch.stack([country[country_index] for country in actuals]).squeeze()

    if loss == "quantile":
        quantile_values = [0.1, 0.25, 0.5, 0.75, 0.9] 
    
        plt.figure(figsize=(14, 8))
        
        plt.style.use('seaborn-v0_8-whitegrid')
        
        # Plot observed values
        plt.plot(dates, country_actuals, label="Observed", color="#1F77B4", linewidth=2.5, marker='o', markersize=5)

        median_idx = 2  
        plt.plot(dates, country_predictions[:, median_idx], 'o-',
                label=f"Median Prediction (q={quantile_values[median_idx]})",
                color="#D62728", linewidth=2)
        
        quantile_colors = ["#AAAAAA", "#888888", "#D62728", "#888888", "#AAAAAA"]
        for i, q in enumerate(quantile_values):
            if i != median_idx:  
                plt.plot(dates, country_predictions[:, i], '--',
                        label=f"Quantile {q}", alpha=0.7, color=quantile_colors[i])
        
        plt.fill_between(
            dates,
            country_predictions[:, 0],  # 10th percentile
            country_predictions[:, -1],  # 90th percentile
            color="orange",
            alpha=0.3,
            label="80% Prediction Interval"
        )
        

        plt.xlabel("Time Period", fontsize=12, fontweight='bold')
        plt.ylabel(f"{target_var}", fontsize=12, fontweight='bold')
        plt.title(f"{country_name}: Observed vs. Predicted {target_var}", 
                fontsize=16, fontweight='bold', pad=20)
        
        plt.legend(loc='best', frameon=True, framealpha=0.9, fontsize=10)
        
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        plt.tick_params(axis='both', which='major', labelsize=10)
        
        plt.box(True)
        
        plt.tight_layout()
        
        plt.show()
    else:
        plt.figure(figsize=(14, 8))
        plt.plot(dates, country_actuals, label="Observed", color="#1F77B4", linewidth=2.5, marker='o', markersize=5)
        plt.plot(dates, country_predictions, label="Predicted", color="#D62728", linewidth=2)
        plt.xlabel("Time Period", fontsize=12, fontweight='bold')
        plt.ylabel(f"{target_var}", fontsize=12, fontweight='bold')
        plt.title(f"{country_name}: Observed vs. Predicted {target_var}", fontsize=16, fontweight='bold', pad=20)
        
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        plt.tick_params(axis='both', which='major', labelsize=10)
        
        plt.box(True)
        
        plt.tight_layout()
        
        plt.show()

def get_country_index(df, country):
    unique_countries = df['Country_index'].unique()
    country_idx = np.where(unique_countries == country)[0][0]
    return country_idx

def get_country_tft_predicitons(df, predictions, country):
    country_idx = get_country_index(df, country)
    return predictions.prediction[country_idx].detach().cpu().numpy()

def get_country_tft_actual(df, actuals, country):
    country_index = get_country_index(df, country)
    return actuals[country_index]



def get_prediction_index(x, df):
    country_lengths = df.groupby(level='Country').size()
    longest_country = country_lengths.idxmax()
    
    longest_country_df = df.xs(longest_country, level='Country')
    
    prediction_indices = x['decoder_time_idx'][1]
    prediction_indices_cpu = prediction_indices.cpu()

    time_mapping = {}
    for idx, date in zip(longest_country_df['time_idx'], longest_country_df.index.get_level_values('TIME_PERIOD')):
        time_mapping[idx] = date

    prediction_dates = [time_mapping.get(idx.item()) for idx in prediction_indices_cpu if idx.item() in time_mapping]

    result_df = pd.DataFrame({
        'time_idx': prediction_indices_cpu,
        'date': prediction_dates
    })
    print(result_df)
    return result_df['date']

# the test set is equal to the takes the last time in the dataset, 
def get_test_data(df, last_time_for_prediction, max_prediction_length, max_encoder_length, training, batch_size):
    
    test = df[(df['time_idx'] >= (last_time_for_prediction - max_prediction_length- max_encoder_length)) & (df['time_idx'] <= last_time_for_prediction)]

    testing = TimeSeriesDataSet.from_dataset(
    training, test, predict=True, stop_randomization=True
    )

    test_dataloader = testing.to_dataloader(
        train=False, batch_size=batch_size * 10, num_workers=0
    )
    return test_dataloader


import Results.tools.helpers as hp
def get_test_data_unbalanced(df, max_prediction_length, max_encoder_length, training, batch_size):
    test_data = []
    
    for country in df.index.get_level_values('Country').unique():
        country_df = hp.get_country(df, country)
        country_last_time = country_df['time_idx'].max()
        
        country_test = country_df[
            (country_df['time_idx'] >= (country_last_time - max_prediction_length - max_encoder_length)) & 
            (country_df['time_idx'] <= country_last_time)
        ]
        test_data.append(country_test)
    
    test = pd.concat(test_data)
    
    testing = TimeSeriesDataSet.from_dataset(
        training, test, predict=True, stop_randomization=True
    )

    test_dataloader = testing.to_dataloader(
        train=False, batch_size=batch_size * 10, num_workers=0
    )
    return test_dataloader




