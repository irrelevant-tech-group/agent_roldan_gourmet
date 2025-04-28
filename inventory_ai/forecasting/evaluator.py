"""
Forecasting Evaluator Module.

This module provides functionality to evaluate and compare
different forecasting models using various metrics.
"""

import logging
from typing import Dict, List, Union, Optional, Any

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

from inventory_ai.forecasting.models import ForecastModel

logger = logging.getLogger(__name__)


class ForecastEvaluator:
    """
    A class for evaluating and comparing forecasting models.
    
    This class provides methods for testing forecasting models on
    historical data and selecting the best model based on accuracy metrics.
    """
    
    def __init__(self):
        """Initialize the ForecastEvaluator."""
        pass
    
    def evaluate_model(self, 
                      model: ForecastModel, 
                      test_data: pd.DataFrame,
                      forecast_horizon: int = 3) -> Dict[str, Any]:
        """
        Evaluate a forecasting model on test data.
        
        Args:
            model (ForecastModel): The forecasting model to evaluate.
            test_data (pd.DataFrame): Historical data for testing.
            forecast_horizon (int, optional): Number of periods to forecast. Defaults to 3.
            
        Returns:
            Dict[str, Any]: Dictionary with evaluation metrics.
        """
        if not hasattr(model, 'fitted') or not model.fitted:
            raise ValueError("Model has not been fitted. Call fit() before evaluation.")
            
        try:
            # Get forecasts from the model
            forecasts = model.predict(horizon=forecast_horizon)
            
            if forecasts.empty:
                logger.warning("Model produced empty forecasts, cannot evaluate")
                return {
                    'model_type': model.get_model_info()['model_type'],
                    'error': 'Empty forecasts'
                }
            
            # Prepare the actual values for comparison
            # Group by SKU and get the most recent `forecast_horizon` months
            sku_groups = test_data.sort_values('Date').groupby('Product Code (SKU)')
            
            actual_values = []
            for sku, group in sku_groups:
                # Take the last `forecast_horizon` data points
                recent_data = group.tail(forecast_horizon)
                
                for _, row in recent_data.iterrows():
                    actual_values.append({
                        'SKU': sku,
                        'Date': row['Date'],
                        'Actual Units': row['Units Sold']
                    })
            
            actual_df = pd.DataFrame(actual_values)
            
            # Match forecasts with actuals
            merged = pd.merge(
                forecasts,
                actual_df,
                left_on=['SKU', 'Forecast Date'],
                right_on=['SKU', 'Date'],
                how='inner'
            )
            
            if merged.empty:
                logger.warning("No matching data points found for evaluation")
                return {
                    'model_type': model.get_model_info()['model_type'],
                    'error': 'No matching data points'
                }
            
            # Calculate error metrics
            mae = mean_absolute_error(merged['Actual Units'], merged['Forecasted Units'])
            rmse = np.sqrt(mean_squared_error(merged['Actual Units'], merged['Forecasted Units']))
            
            # Calculate MAPE (Mean Absolute Percentage Error)
            # Avoid division by zero
            mask = merged['Actual Units'] != 0
            mape = np.mean(np.abs((merged.loc[mask, 'Actual Units'] - 
                                   merged.loc[mask, 'Forecasted Units']) / 
                                   merged.loc[mask, 'Actual Units'])) * 100
            
            # Calculate accuracy per SKU
            sku_metrics = {}
            for sku, group in merged.groupby('SKU'):
                sku_mae = mean_absolute_error(group['Actual Units'], group['Forecasted Units'])
                sku_metrics[sku] = {'MAE': sku_mae}
            
            return {
                'model_type': model.get_model_info()['model_type'],
                'overall_mae': mae,
                'overall_rmse': rmse,
                'overall_mape': mape,
                'data_points': len(merged),
                'metrics_by_sku': sku_metrics
            }
            
        except Exception as e:
            logger.error(f"Error evaluating model: {str(e)}")
            raise
    
    def compare_models(self, 
                      models: List[ForecastModel], 
                      test_data: pd.DataFrame,
                      forecast_horizon: int = 3) -> Dict[str, Any]:
        """
        Compare multiple forecasting models on test data.
        
        Args:
            models (List[ForecastModel]): List of forecasting models to compare.
            test_data (pd.DataFrame): Historical data for testing.
            forecast_horizon (int, optional): Number of periods to forecast. Defaults to 3.
            
        Returns:
            Dict[str, Any]: Dictionary with comparison results.
        """
        results = {}
        best_model = None
        best_mae = float('inf')
        
        try:
            for model in models:
                model_name = model.get_model_info()['model_type']
                logger.info(f"Evaluating model: {model_name}")
                
                eval_result = self.evaluate_model(model, test_data, forecast_horizon)
                results[model_name] = eval_result
                
                # Check if this is the best model so far
                if 'overall_mae' in eval_result and eval_result['overall_mae'] < best_mae:
                    best_mae = eval_result['overall_mae']
                    best_model = model_name
            
            return {
                'models': results,
                'best_model': best_model,
                'comparison_metric': 'MAE'
            }
            
        except Exception as e:
            logger.error(f"Error comparing models: {str(e)}")
            raise
    
    def select_best_model_per_sku(self, 
                                 models: List[ForecastModel], 
                                 test_data: pd.DataFrame,
                                 forecast_horizon: int = 3) -> Dict[str, str]:
        """
        Select the best forecasting model for each SKU.
        
        Args:
            models (List[ForecastModel]): List of forecasting models to compare.
            test_data (pd.DataFrame): Historical data for testing.
            forecast_horizon (int, optional): Number of periods to forecast. Defaults to 3.
            
        Returns:
            Dict[str, str]: Dictionary mapping SKUs to their best model.
        """
        try:
            # Get all SKUs from test data
            skus = test_data['Product Code (SKU)'].unique()
            
            # Compare all models
            comparison = self.compare_models(models, test_data, forecast_horizon)
            
            # For each SKU, find the model with lowest MAE
            best_models = {}
            for sku in skus:
                best_model_name = None
                best_mae = float('inf')
                
                for model_name, results in comparison['models'].items():
                    if 'metrics_by_sku' in results and sku in results['metrics_by_sku']:
                        sku_mae = results['metrics_by_sku'][sku]['MAE']
                        if sku_mae < best_mae:
                            best_mae = sku_mae
                            best_model_name = model_name
                
                if best_model_name:
                    best_models[sku] = best_model_name
                else:
                    # Default to the overall best model if no specific data for this SKU
                    best_models[sku] = comparison.get('best_model', models[0].get_model_info()['model_type'])
            
            return best_models
            
        except Exception as e:
            logger.error(f"Error selecting best models per SKU: {str(e)}")
            raise
    
    def cross_validate(self, 
                      model: ForecastModel,
                      data: pd.DataFrame,
                      num_folds: int = 3,
                      forecast_horizon: int = 3) -> Dict[str, Any]:
        """
        Perform time series cross-validation on a model.
        
        Args:
            model (ForecastModel): The forecasting model to validate.
            data (pd.DataFrame): Historical data for validation.
            num_folds (int, optional): Number of validation folds. Defaults to 3.
            forecast_horizon (int, optional): Number of periods to forecast. Defaults to 3.
            
        Returns:
            Dict[str, Any]: Cross-validation results.
        """
        if data.empty:
            logger.warning("Empty data provided for cross-validation")
            return {'error': 'Empty data'}
        
        try:
            # Sort data by date
            data = data.sort_values('Date')
            
            # Calculate the size of each fold
            total_periods = len(data['Date'].unique())
            fold_size = total_periods // (num_folds + 1)
            
            if fold_size < forecast_horizon:
                logger.warning("Not enough data for reliable cross-validation")
                return {'error': 'Insufficient data for cross-validation'}
            
            results = []
            
            # Perform rolling-window validation
            for i in range(num_folds):
                # Calculate cutoff dates for this fold
                cutoff_date = data['Date'].unique()[-(i+1)*fold_size]
                
                # Split data
                train_data = data[data['Date'] < cutoff_date]
                test_data = data[(data['Date'] >= cutoff_date) & 
                               (data['Date'] < pd.Timestamp(cutoff_date) + pd.DateOffset(months=forecast_horizon))]
                
                if train_data.empty or test_data.empty:
                    continue
                
                # Fit model on training data
                model.fit(train_data)
                
                # Evaluate on test data
                fold_result = self.evaluate_model(model, test_data, forecast_horizon)
                fold_result['fold'] = i + 1
                fold_result['train_cutoff_date'] = cutoff_date
                
                results.append(fold_result)
            
            # Calculate average metrics across folds
            if not results:
                return {'error': 'No valid folds for cross-validation'}
            
            avg_mae = np.mean([r['overall_mae'] for r in results if 'overall_mae' in r])
            avg_rmse = np.mean([r['overall_rmse'] for r in results if 'overall_rmse' in r])
            avg_mape = np.mean([r['overall_mape'] for r in results if 'overall_mape' in r])
            
            return {
                'model_type': model.get_model_info()['model_type'],
                'num_folds': len(results),
                'avg_mae': avg_mae,
                'avg_rmse': avg_rmse,
                'avg_mape': avg_mape,
                'fold_results': results
            }
            
        except Exception as e:
            logger.error(f"Error in cross-validation: {str(e)}")
            raise