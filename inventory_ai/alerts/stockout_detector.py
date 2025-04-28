"""
Stockout Detector Module.

This module provides functionality to detect potential stockouts
and generate inventory alerts based on forecasted demand.
"""

import logging
from typing import Dict, List, Union, Optional, Any, Tuple
from datetime import datetime, timedelta

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class StockoutDetector:
    """
    A class for detecting potential stockouts based on inventory and forecasted demand.
    
    This class analyzes current inventory levels against forecasted demand
    to identify products at risk of stockout.
    """
    
    def __init__(self, 
                safety_stock_factor: float = 1.5,
                low_rotation_threshold: int = 3):
        """
        Initialize the StockoutDetector.
        
        Args:
            safety_stock_factor (float, optional): Multiplier for minimum stock requirements
                to determine safety stock levels. Defaults to 1.5.
            low_rotation_threshold (int, optional): Number of months with zero or minimal sales
                to classify as low rotation. Defaults to 3.
        """
        self.safety_stock_factor = safety_stock_factor
        self.low_rotation_threshold = low_rotation_threshold
    
    def detect_stockouts(self, 
                        inventory_df: pd.DataFrame,
                        forecast_df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect potential stockouts based on current inventory and forecasted demand.
        
        Args:
            inventory_df (pd.DataFrame): Current inventory data.
                Should contain 'SKU', 'Current Stock', and 'Minimum Stock' columns.
            forecast_df (pd.DataFrame): Forecasted demand data.
                Should contain 'SKU', 'Forecast Date', and 'Forecasted Units' columns.
                
        Returns:
            pd.DataFrame: DataFrame with stockout risk information.
        """
        if inventory_df.empty or forecast_df.empty:
            logger.warning("Empty data provided for stockout detection")
            return pd.DataFrame()
        
        try:
            # Validate required columns
            inventory_cols = ['SKU', 'Current Stock', 'Minimum Stock']
            forecast_cols = ['SKU', 'Forecast Date', 'Forecasted Units']
            
            for col in inventory_cols:
                if col not in inventory_df.columns:
                    raise ValueError(f"Required column '{col}' not found in inventory data")
            
            for col in forecast_cols:
                if col not in forecast_df.columns:
                    raise ValueError(f"Required column '{col}' not found in forecast data")
            
            # Calculate total forecasted demand for each SKU
            # Group by SKU and sum the forecasted units
            total_forecast = forecast_df.groupby('SKU')['Forecasted Units'].sum().reset_index()
            total_forecast.rename(columns={'Forecasted Units': 'Total Forecasted Demand'}, inplace=True)
            
            # Merge with inventory data
            merged_df = pd.merge(inventory_df, total_forecast, on='SKU', how='left')
            merged_df['Total Forecasted Demand'].fillna(0, inplace=True)
            
            # Calculate safety stock levels
            merged_df['Safety Stock'] = merged_df['Minimum Stock'] * self.safety_stock_factor
            
            # Calculate remaining stock after forecasted demand
            merged_df['Projected Remaining Stock'] = merged_df['Current Stock'] - merged_df['Total Forecasted Demand']
            
            # Determine stockout risk
            merged_df['Stockout Risk'] = 'None'
            
            # High risk: Projected stock will be below minimum stock
            high_risk_mask = merged_df['Projected Remaining Stock'] < merged_df['Minimum Stock']
            merged_df.loc[high_risk_mask, 'Stockout Risk'] = 'High'
            
            # Medium risk: Projected stock will be below safety stock but above minimum
            medium_risk_mask = ((merged_df['Projected Remaining Stock'] >= merged_df['Minimum Stock']) & 
                             (merged_df['Projected Remaining Stock'] < merged_df['Safety Stock']))
            merged_df.loc[medium_risk_mask, 'Stockout Risk'] = 'Medium'
            
            # Low risk: Projected stock will be below current + 20% but above safety stock
            low_risk_mask = ((merged_df['Projected Remaining Stock'] >= merged_df['Safety Stock']) & 
                          (merged_df['Projected Remaining Stock'] < merged_df['Current Stock'] * 1.2))
            merged_df.loc[low_risk_mask, 'Stockout Risk'] = 'Low'
            
            # Calculate days until stockout (approximate)
            # Assuming average daily demand = monthly forecast / 30
            merged_df['Avg Daily Demand'] = merged_df['Total Forecasted Demand'] / 90  # Assuming 3-month forecast
            
            # Avoid division by zero
            merged_df['Days Until Stockout'] = np.where(
                merged_df['Avg Daily Demand'] > 0,
                merged_df['Current Stock'] / merged_df['Avg Daily Demand'],
                999  # Large number for items with no demand
            )
            
            # Clean up and prepare the final output
            result_columns = [
                'SKU', 'Product', 'Current Stock', 'Minimum Stock', 'Safety Stock',
                'Total Forecasted Demand', 'Projected Remaining Stock', 
                'Stockout Risk', 'Days Until Stockout'
            ]
            
            # Handle if some Product column is missing
            if 'Product' not in merged_df.columns:
                merged_df['Product'] = merged_df['SKU']
                
            result_df = merged_df[result_columns].copy()
            
            # Sort by stockout risk priority
            risk_order = {'High': 0, 'Medium': 1, 'Low': 2, 'None': 3}
            result_df['Risk Priority'] = result_df['Stockout Risk'].map(risk_order)
            result_df = result_df.sort_values('Risk Priority').drop('Risk Priority', axis=1)
            
            logger.info(f"Stockout detection completed. Found {sum(result_df['Stockout Risk'] != 'None')} items at risk")
            return result_df
            
        except Exception as e:
            logger.error(f"Error detecting stockouts: {str(e)}")
            raise
    
    def identify_low_rotation_items(self, sales_df: pd.DataFrame, months: int = 3) -> pd.DataFrame:
        """
        Identify low-rotation inventory items based on sales history.
        
        Args:
            sales_df (pd.DataFrame): Historical sales data.
                Should contain 'Product Code (SKU)', 'Date', and 'Units Sold' columns.
            months (int, optional): Number of recent months to consider. Defaults to 3.
                
        Returns:
            pd.DataFrame: DataFrame with low-rotation items.
        """
        if sales_df.empty:
            logger.warning("Empty sales data provided for low rotation analysis")
            return pd.DataFrame()
            
        try:
            # Validate required columns
            required_cols = ['Product Code (SKU)', 'Date', 'Units Sold']
            for col in required_cols:
                if col not in sales_df.columns:
                    raise ValueError(f"Required column '{col}' not found in sales data")
            
            # Ensure Date column is datetime
            if not pd.api.types.is_datetime64_any_dtype(sales_df['Date']):
                sales_df['Date'] = pd.to_datetime(sales_df['Date'], errors='coerce')
            
            # Filter for recent data
            cutoff_date = pd.Timestamp.now() - pd.DateOffset(months=months)
            recent_sales = sales_df[sales_df['Date'] >= cutoff_date].copy()
            
            if recent_sales.empty:
                logger.warning(f"No sales data found within the last {months} months")
                return pd.DataFrame()
            
            # Calculate total sales per SKU
            sales_by_sku = recent_sales.groupby('Product Code (SKU)')['Units Sold'].sum().reset_index()
            
            # Identify low rotation items (items with zero or minimal sales)
            low_rotation = sales_by_sku[sales_by_sku['Units Sold'] <= self.low_rotation_threshold].copy()
            
            # Get product names if available
            if 'Product Name' in recent_sales.columns:
                product_names = recent_sales.groupby('Product Code (SKU)')['Product Name'].first().reset_index()
                low_rotation = pd.merge(low_rotation, product_names, on='Product Code (SKU)', how='left')
            
            # Add rotation status and period
            low_rotation['Rotation Status'] = 'Low'
            low_rotation['Analysis Period'] = f"Last {months} months"
            
            # Calculate percentage of total SKUs
            total_skus = len(sales_by_sku)
            percentage = (len(low_rotation) / total_skus * 100) if total_skus > 0 else 0
            logger.info(f"Identified {len(low_rotation)} low rotation items ({percentage:.1f}% of total)")
            
            return low_rotation
            
        except Exception as e:
            logger.error(f"Error identifying low rotation items: {str(e)}")
            raise
    
    def generate_replenishment_recommendations(self, 
                                              inventory_df: pd.DataFrame,
                                              forecast_df: pd.DataFrame,
                                              sales_df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate replenishment recommendations based on stockout risk and item rotation.
        
        Args:
            inventory_df (pd.DataFrame): Current inventory data.
            forecast_df (pd.DataFrame): Forecasted demand data.
            sales_df (pd.DataFrame): Historical sales data.
                
        Returns:
            pd.DataFrame: DataFrame with replenishment recommendations.
        """
        try:
            # Detect stockout risks
            stockout_risks = self.detect_stockouts(inventory_df, forecast_df)
            
            # Identify low rotation items
            low_rotation_items = self.identify_low_rotation_items(sales_df)
            low_rotation_skus = set(low_rotation_items['Product Code (SKU)']) if not low_rotation_items.empty else set()
            
            # Prepare recommendations
            if stockout_risks.empty:
                logger.warning("No stockout risk data available for recommendations")
                return pd.DataFrame()
            
            # Copy data for manipulation
            recommendations = stockout_risks.copy()
            
            # Add recommendations based on risk level
            recommendations['Replenishment Priority'] = 'None'
            recommendations['Recommended Action'] = 'No action needed'
            recommendations['Order Quantity'] = 0
            
            # High risk items
            high_risk_mask = recommendations['Stockout Risk'] == 'High'
            low_rotation_mask = recommendations['SKU'].isin(low_rotation_skus)
            
            # High risk but not low rotation: Immediate replenishment
            high_risk_not_low_rotation = high_risk_mask & ~low_rotation_mask
            recommendations.loc[high_risk_not_low_rotation, 'Replenishment Priority'] = 'High'
            recommendations.loc[high_risk_not_low_rotation, 'Recommended Action'] = 'Immediate replenishment'
            
            # Calculate recommended order quantity: target safety stock level + forecast - current
            recommendations.loc[high_risk_not_low_rotation, 'Order Quantity'] = (
                recommendations.loc[high_risk_not_low_rotation, 'Safety Stock'] + 
                recommendations.loc[high_risk_not_low_rotation, 'Total Forecasted Demand'] - 
                recommendations.loc[high_risk_not_low_rotation, 'Current Stock']
            ).apply(lambda x: max(0, round(x)))
            
            # Medium risk but not low rotation: Plan replenishment soon
            medium_risk_not_low_rotation = (recommendations['Stockout Risk'] == 'Medium') & ~low_rotation_mask
            recommendations.loc[medium_risk_not_low_rotation, 'Replenishment Priority'] = 'Medium'
            recommendations.loc[medium_risk_not_low_rotation, 'Recommended Action'] = 'Plan replenishment within 2 weeks'
            recommendations.loc[medium_risk_not_low_rotation, 'Order Quantity'] = (
                recommendations.loc[medium_risk_not_low_rotation, 'Safety Stock'] - 
                recommendations.loc[medium_risk_not_low_rotation, 'Projected Remaining Stock']
            ).apply(lambda x: max(0, round(x)))
            
            # Low risk but not low rotation: Consider replenishment
            low_risk_not_low_rotation = (recommendations['Stockout Risk'] == 'Low') & ~low_rotation_mask
            recommendations.loc[low_risk_not_low_rotation, 'Replenishment Priority'] = 'Low'
            recommendations.loc[low_risk_not_low_rotation, 'Recommended Action'] = 'Consider including in next order'
            recommendations.loc[low_risk_not_low_rotation, 'Order Quantity'] = (
                recommendations.loc[low_risk_not_low_rotation, 'Safety Stock'] - 
                recommendations.loc[low_risk_not_low_rotation, 'Projected Remaining Stock']
            ).apply(lambda x: max(0, round(x)))
            
            # Low rotation items: Special handling regardless of risk
            recommendations.loc[low_rotation_mask, 'Replenishment Priority'] = 'Low-Rotation'
            recommendations.loc[low_rotation_mask, 'Recommended Action'] = 'Evaluate before replenishing'
            recommendations.loc[low_rotation_mask & high_risk_mask, 'Order Quantity'] = (
                recommendations.loc[low_rotation_mask & high_risk_mask, 'Minimum Stock'] - 
                recommendations.loc[low_rotation_mask & high_risk_mask, 'Projected Remaining Stock']
            ).apply(lambda x: max(0, round(x)))
            
            # Ensure we don't recommend tiny orders
            min_order_threshold = 3  # Minimum reasonable order quantity
            recommendations.loc[recommendations['Order Quantity'] < min_order_threshold, 'Order Quantity'] = 0
            recommendations.loc[recommendations['Order Quantity'] == 0, 'Recommended Action'] = 'No replenishment needed'
            
            # Sort by priority
            priority_order = {'High': 0, 'Medium': 1, 'Low': 2, 'Low-Rotation': 3, 'None': 4}
            recommendations['Priority Rank'] = recommendations['Replenishment Priority'].map(priority_order)
            recommendations = recommendations.sort_values('Priority Rank').drop('Priority Rank', axis=1)
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating replenishment recommendations: {str(e)}")
            raise