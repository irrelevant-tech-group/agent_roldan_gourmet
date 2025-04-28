"""
Forecasting Models Module.

This module provides demand forecasting functionality using various
time series forecasting methods.
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Union, Optional, Any, Tuple

import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.api import SimpleExpSmoothing
from statsmodels.tsa.arima.model import ARIMA

logger = logging.getLogger(__name__)


class ForecastModel(ABC):
    """
    Abstract base class for all forecasting models.
    
    This class defines the interface that all forecasting models must implement,
    allowing for easy interchangeability of different models.
    """
    
    @abstractmethod
    def fit(self, data: pd.DataFrame) -> None:
        """
        Fit the model on historical data.
        
        Args:
            data (pd.DataFrame): Historical data to train the model on.
        """
        pass
    
    @abstractmethod
    def predict(self, horizon: int) -> pd.DataFrame:
        """
        Generate forecasts for the specified time horizon.
        
        Args:
            horizon (int): Number of time periods to forecast ahead.
            
        Returns:
            pd.DataFrame: DataFrame containing forecasted values.
        """
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the model.
        
        Returns:
            Dict[str, Any]: Dictionary containing model information.
        """
        pass


class MovingAverageModel(ForecastModel):
    """
    Simple Moving Average forecasting model.
    
    This model calculates forecasts based on the average of
    the most recent n time periods.
    """
    
    def __init__(self, window_size: int = 3):
        """
        Initialize the Moving Average model.
        
        Args:
            window_size (int, optional): Size of the moving window. Defaults to 3.
        """
        self.window_size = window_size
        self.history = None
        self.product_histories = {}
        self.fitted = False
    
    def fit(self, data: pd.DataFrame) -> None:
        """
        Fit the Moving Average model on historical data.
        
        Args:
            data (pd.DataFrame): Historical sales data.
                Should contain 'Product Code (SKU)', 'Date', and 'Units Sold' columns.
        """
        if data.empty:
            logger.warning("Received empty data for fitting MovingAverageModel")
            return
        
        try:
            required_columns = ['Product Code (SKU)', 'Date', 'Units Sold']
            for col in required_columns:
                if col not in data.columns:
                    raise ValueError(f"Required column '{col}' not found in data")
            
            # Store a copy of the data
            self.history = data.copy()
            
            # Group by product and sort by date
            grouped = data.sort_values('Date').groupby('Product Code (SKU)')
            
            # Store time series for each product
            self.product_histories = {}
            for sku, group in grouped:
                # Convert to time series with date index
                ts = group.set_index('Date')['Units Sold']
                # Resample to monthly frequency and fill missing values
                monthly_ts = ts.resample('M').sum().fillna(0)
                self.product_histories[sku] = monthly_ts
            
            self.fitted = True
            logger.info(f"MovingAverageModel fitted on {len(self.product_histories)} products")
            
        except Exception as e:
            logger.error(f"Error fitting MovingAverageModel: {str(e)}")
            raise
    
    def predict(self, horizon: int = 3) -> pd.DataFrame:
        """
        Generate moving average forecasts for all products.
        
        Args:
            horizon (int, optional): Number of months to forecast. Defaults to 3.
            
        Returns:
            pd.DataFrame: DataFrame with forecasts for each product and month.
        """
        if not self.fitted:
            raise ValueError("Model has not been fitted. Call fit() before predict().")
        
        if not self.product_histories:
            logger.warning("No product histories available for prediction")
            return pd.DataFrame()
        
        forecasts = []
        
        try:
            for sku, ts in self.product_histories.items():
                # Calculate the moving average of the last window_size periods
                if len(ts) >= self.window_size:
                    forecast_value = ts[-self.window_size:].mean()
                else:
                    # If we have less data than window_size, use whatever we have
                    forecast_value = ts.mean() if len(ts) > 0 else 0
                
                # Create forecast for each month in the horizon
                last_date = ts.index[-1] if not ts.empty else pd.Timestamp.now()
                
                for i in range(1, horizon + 1):
                    # Calculate the next month
                    next_month = last_date + pd.DateOffset(months=i)
                    
                    forecasts.append({
                        'SKU': sku,
                        'Forecast Date': next_month,
                        'Forecasted Units': round(forecast_value),
                        'Forecast Period': f'{next_month.month_name()} {next_month.year}',
                        'Model': 'Moving Average'
                    })
            
            forecast_df = pd.DataFrame(forecasts)
            return forecast_df
            
        except Exception as e:
            logger.error(f"Error generating moving average forecasts: {str(e)}")
            raise
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the moving average model.
        
        Returns:
            Dict[str, Any]: Dictionary with model information.
        """
        return {
            'model_type': 'Moving Average',
            'window_size': self.window_size,
            'products_fitted': len(self.product_histories) if self.product_histories else 0,
            'fitted': self.fitted
        }


class ExponentialSmoothingModel(ForecastModel):
    """
    Exponential Smoothing forecasting model.
    
    This model gives more weight to recent observations while
    still considering older observations with exponentially decreasing weights.
    """
    
    def __init__(self, smoothing_level: float = 0.3):
        """
        Initialize the Exponential Smoothing model.
        
        Args:
            smoothing_level (float, optional): Smoothing factor between 0 and 1.
                Higher values give more weight to recent observations. Defaults to 0.3.
        """
        self.smoothing_level = smoothing_level
        self.history = None
        self.product_histories = {}
        self.fitted_models = {}
        self.fitted = False
    
    def fit(self, data: pd.DataFrame) -> None:
        """
        Fit the Exponential Smoothing model on historical data.
        
        Args:
            data (pd.DataFrame): Historical sales data.
                Should contain 'Product Code (SKU)', 'Date', and 'Units Sold' columns.
        """
        if data.empty:
            logger.warning("Received empty data for fitting ExponentialSmoothingModel")
            return

        try:
            required_columns = ['Product Code (SKU)', 'Date', 'Units Sold']
            for col in required_columns:
                if col not in data.columns:
                    raise ValueError(f"Required column '{col}' not found in data")
            
            # Store a copy of the data
            self.history = data.copy()
            
            # Group by product and sort by date
            grouped = data.sort_values('Date').groupby('Product Code (SKU)')
            
            # Store time series and fit models for each product
            self.product_histories = {}
            self.fitted_models = {}
            
            for sku, group in grouped:
                # Convert to time series with date index
                ts = group.set_index('Date')['Units Sold']
                # Resample to monthly frequency and fill missing values
                monthly_ts = ts.resample('M').sum().fillna(0)
                self.product_histories[sku] = monthly_ts
                
                # Only fit if we have enough data
                if len(monthly_ts) >= 4:  # Need at least some data points
                    try:
                        # Fit simple exponential smoothing model
                        model = SimpleExpSmoothing(monthly_ts)
                        fitted_model = model.fit(smoothing_level=self.smoothing_level, optimized=False)
                        self.fitted_models[sku] = fitted_model
                    except Exception as e:
                        logger.warning(f"Could not fit exponential smoothing model for SKU {sku}: {str(e)}")
            
            self.fitted = True
            logger.info(f"ExponentialSmoothingModel fitted on {len(self.fitted_models)} products")
            
        except Exception as e:
            logger.error(f"Error fitting ExponentialSmoothingModel: {str(e)}")
            raise
    
    def predict(self, horizon: int = 3) -> pd.DataFrame:
        """
        Generate exponential smoothing forecasts for all products.
        
        Args:
            horizon (int, optional): Number of months to forecast. Defaults to 3.
            
        Returns:
            pd.DataFrame: DataFrame with forecasts for each product and month.
        """
        if not self.fitted:
            raise ValueError("Model has not been fitted. Call fit() before predict().")
        
        if not self.product_histories:
            logger.warning("No product histories available for prediction")
            return pd.DataFrame()
        
        forecasts = []
        
        try:
            for sku, ts in self.product_histories.items():
                # Get the last date in the time series
                last_date = ts.index[-1] if not ts.empty else pd.Timestamp.now()
                
                if sku in self.fitted_models:
                    # Use the fitted model to make forecasts
                    model_forecast = self.fitted_models[sku].forecast(horizon)
                    
                    for i in range(horizon):
                        next_date = model_forecast.index[i]
                        forecasts.append({
                            'SKU': sku,
                            'Forecast Date': next_date,
                            'Forecasted Units': round(max(0, model_forecast.iloc[i])),
                            'Forecast Period': f'{next_date.month_name()} {next_date.year}',
                            'Model': 'Exponential Smoothing'
                        })
                else:
                    # Fall back to a simple average if no model was fitted
                    forecast_value = ts.mean() if len(ts) > 0 else 0
                    
                    for i in range(1, horizon + 1):
                        next_month = last_date + pd.DateOffset(months=i)
                        forecasts.append({
                            'SKU': sku,
                            'Forecast Date': next_month,
                            'Forecasted Units': round(forecast_value),
                            'Forecast Period': f'{next_month.month_name()} {next_month.year}',
                            'Model': 'Average (fallback)'
                        })
            
            forecast_df = pd.DataFrame(forecasts)
            return forecast_df
            
        except Exception as e:
            logger.error(f"Error generating exponential smoothing forecasts: {str(e)}")
            raise
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the exponential smoothing model.
        
        Returns:
            Dict[str, Any]: Dictionary with model information.
        """
        return {
            'model_type': 'Exponential Smoothing',
            'smoothing_level': self.smoothing_level,
            'products_fitted': len(self.fitted_models),
            'fitted': self.fitted
        }


class HoltWintersModel(ForecastModel):
    """
    Holt-Winters forecasting model with seasonality.
    
    This model extends exponential smoothing with trend and
    seasonality components for more accurate forecasts.
    """
    
    def __init__(self, 
                seasonal_periods: int = 12, 
                trend: str = 'add', 
                seasonal: str = 'add',
                smoothing_level: float = 0.3,
                smoothing_trend: float = 0.1,
                smoothing_seasonal: float = 0.1):
        """
        Initialize the Holt-Winters model.
        
        Args:
            seasonal_periods (int, optional): Number of periods in a season. Defaults to 12 (months).
            trend (str, optional): Type of trend component. Defaults to 'add'.
            seasonal (str, optional): Type of seasonal component. Defaults to 'add'.
            smoothing_level (float, optional): Level smoothing parameter. Defaults to 0.3.
            smoothing_trend (float, optional): Trend smoothing parameter. Defaults to 0.1.
            smoothing_seasonal (float, optional): Seasonal smoothing parameter. Defaults to 0.1.
        """
        self.seasonal_periods = seasonal_periods
        self.trend = trend
        self.seasonal = seasonal
        self.smoothing_level = smoothing_level
        self.smoothing_trend = smoothing_trend
        self.smoothing_seasonal = smoothing_seasonal
        self.history = None
        self.product_histories = {}
        self.fitted_models = {}
        self.fitted = False
    
    def fit(self, data: pd.DataFrame) -> None:
        """
        Fit the Holt-Winters model on historical data.
        
        Args:
            data (pd.DataFrame): Historical sales data.
                Should contain 'Product Code (SKU)', 'Date', and 'Units Sold' columns.
        """
        if data.empty:
            logger.warning("Received empty data for fitting HoltWintersModel")
            return
        
        try:
            required_columns = ['Product Code (SKU)', 'Date', 'Units Sold']
            for col in required_columns:
                if col not in data.columns:
                    raise ValueError(f"Required column '{col}' not found in data")
            
            # Store a copy of the data
            self.history = data.copy()
            
            # Group by product and sort by date
            grouped = data.sort_values('Date').groupby('Product Code (SKU)')
            
            # Store time series and fit models for each product
            self.product_histories = {}
            self.fitted_models = {}
            
            for sku, group in grouped:
                # Convert to time series with date index
                ts = group.set_index('Date')['Units Sold']
                # Resample to monthly frequency and fill missing values
                monthly_ts = ts.resample('M').sum().fillna(0)
                self.product_histories[sku] = monthly_ts
                
                # Only fit if we have enough data
                if len(monthly_ts) >= max(2 * self.seasonal_periods, 10):
                    try:
                        # Fit Holt-Winters model
                        model = ExponentialSmoothing(
                            monthly_ts,
                            trend=self.trend,
                            seasonal=self.seasonal,
                            seasonal_periods=self.seasonal_periods
                        )
                        
                        fitted_model = model.fit(
                            smoothing_level=self.smoothing_level,
                            smoothing_trend=self.smoothing_trend,
                            smoothing_seasonal=self.smoothing_seasonal,
                            optimized=False
                        )
                        
                        self.fitted_models[sku] = fitted_model
                    except Exception as e:
                        logger.warning(f"Could not fit Holt-Winters model for SKU {sku}: {str(e)}")
            
            self.fitted = True
            logger.info(f"HoltWintersModel fitted on {len(self.fitted_models)} products")
            
        except Exception as e:
            logger.error(f"Error fitting HoltWintersModel: {str(e)}")
            raise
    
    def predict(self, horizon: int = 3) -> pd.DataFrame:
        """
        Generate Holt-Winters forecasts for all products.
        
        Args:
            horizon (int, optional): Number of months to forecast. Defaults to 3.
            
        Returns:
            pd.DataFrame: DataFrame with forecasts for each product and month.
        """
        if not self.fitted:
            raise ValueError("Model has not been fitted. Call fit() before predict().")
        
        if not self.product_histories:
            logger.warning("No product histories available for prediction")
            return pd.DataFrame()
        
        forecasts = []
        
        try:
            for sku, ts in self.product_histories.items():
                # Get the last date in the time series
                last_date = ts.index[-1] if not ts.empty else pd.Timestamp.now()
                
                if sku in self.fitted_models:
                    # Use the fitted model to make forecasts
                    model_forecast = self.fitted_models[sku].forecast(horizon)
                    
                    for i in range(horizon):
                        next_date = model_forecast.index[i]
                        forecasts.append({
                            'SKU': sku,
                            'Forecast Date': next_date,
                            'Forecasted Units': round(max(0, model_forecast.iloc[i])),
                            'Forecast Period': f'{next_date.month_name()} {next_date.year}',
                            'Model': 'Holt-Winters'
                        })
                else:
                    # Fall back to a simple average if no model was fitted
                    forecast_value = ts.mean() if len(ts) > 0 else 0
                    
                    for i in range(1, horizon + 1):
                        next_month = last_date + pd.DateOffset(months=i)
                        forecasts.append({
                            'SKU': sku,
                            'Forecast Date': next_month,
                            'Forecasted Units': round(forecast_value),
                            'Forecast Period': f'{next_month.month_name()} {next_month.year}',
                            'Model': 'Average (fallback)'
                        })
            
            forecast_df = pd.DataFrame(forecasts)
            return forecast_df
            
        except Exception as e:
            logger.error(f"Error generating Holt-Winters forecasts: {str(e)}")
            raise
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the Holt-Winters model.
        
        Returns:
            Dict[str, Any]: Dictionary with model information.
        """
        return {
            'model_type': 'Holt-Winters',
            'seasonal_periods': self.seasonal_periods,
            'trend': self.trend,
            'seasonal': self.seasonal,
            'smoothing_level': self.smoothing_level,
            'smoothing_trend': self.smoothing_trend,
            'smoothing_seasonal': self.smoothing_seasonal,
            'products_fitted': len(self.fitted_models),
            'fitted': self.fitted
        }