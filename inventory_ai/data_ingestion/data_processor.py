"""
Data Processor Module.

This module provides functionality to clean, validate, and transform
raw data from Google Sheets into structured format for analysis.
"""

import logging
import re
from datetime import datetime
from typing import Dict, List, Union, Optional, Any

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class DataProcessor:
    """
    A processor class for cleaning, validating, and transforming data.
    
    This class handles the transformation of raw data from Google Sheets
    into structured, cleaned data suitable for analysis and forecasting.
    """
    
    def __init__(self):
        """Initialize the DataProcessor."""
        pass
    
    def process_inventory_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process and clean inventory data.
        
        Args:
            df (pd.DataFrame): Raw inventory data from Google Sheets.
        
        Returns:
            pd.DataFrame: Cleaned and processed inventory DataFrame.
        """
        if df.empty:
            logger.warning("Empty inventory data received")
            return df
        
        # Make a copy to avoid modifying the original DataFrame
        processed_df = df.copy()
        
        try:
            # Ensure required columns exist
            required_columns = ['SKU', 'Product', 'Current Stock', 'Minimum Stock', 'Last Update Date']
            self._validate_columns(processed_df, required_columns)
            
            # Clean column names (remove extra spaces, lowercase)
            processed_df.columns = [col.strip() for col in processed_df.columns]
            
            # Convert numeric columns to appropriate types
            if 'Current Stock' in processed_df.columns:
                processed_df['Current Stock'] = pd.to_numeric(processed_df['Current Stock'], errors='coerce')
                processed_df['Current Stock'].fillna(0, inplace=True)
                processed_df['Current Stock'] = processed_df['Current Stock'].astype(int)
            
            if 'Minimum Stock' in processed_df.columns:
                processed_df['Minimum Stock'] = pd.to_numeric(processed_df['Minimum Stock'], errors='coerce')
                processed_df['Minimum Stock'].fillna(0, inplace=True)
                processed_df['Minimum Stock'] = processed_df['Minimum Stock'].astype(int)
            
            # Parse dates
            if 'Last Update Date' in processed_df.columns:
                processed_df['Last Update Date'] = self._parse_dates(processed_df['Last Update Date'])
            
            # Ensure SKU is string type and used as index
            if 'SKU' in processed_df.columns:
                processed_df['SKU'] = processed_df['SKU'].astype(str)
                
            # Remove any duplicate SKUs (keeping the first occurrence)
            if processed_df.duplicated(subset=['SKU']).any():
                logger.warning(f"Found {processed_df.duplicated(subset=['SKU']).sum()} duplicate SKUs in inventory data")
                processed_df.drop_duplicates(subset=['SKU'], keep='first', inplace=True)
            
            logger.info(f"Successfully processed inventory data: {len(processed_df)} records")
            return processed_df
        
        except Exception as e:
            logger.error(f"Error processing inventory data: {str(e)}")
            raise
    
    def process_sales_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process and clean sales data.
        
        Args:
            df (pd.DataFrame): Raw sales data from Google Sheets.
        
        Returns:
            pd.DataFrame: Cleaned and processed sales DataFrame.
        """
        if df.empty:
            logger.warning("Empty sales data received")
            return df
        
        # Make a copy to avoid modifying the original DataFrame
        processed_df = df.copy()
        
        try:
            # Ensure required columns exist
            required_columns = ['Product Code (SKU)', 'Units Sold', 'Date', 'Invoice ID']
            self._validate_columns(processed_df, required_columns)
            
            # Clean column names (remove extra spaces)
            processed_df.columns = [col.strip() for col in processed_df.columns]
            
            # Convert SKU to string
            processed_df['Product Code (SKU)'] = processed_df['Product Code (SKU)'].astype(str)
            
            # Convert numeric columns
            numeric_columns = ['Units Sold', 'Gross Value', 'Discount', 'Subtotal', 
                               'Tax Charge', 'Tax Retention', 'Total']
            
            for column in numeric_columns:
                if column in processed_df.columns:
                    processed_df[column] = pd.to_numeric(processed_df[column], errors='coerce')
                    processed_df[column].fillna(0, inplace=True)
            
            # Convert integer columns
            int_columns = ['Units Sold', 'Invoice ID', 'Tax Retention']
            for column in int_columns:
                if column in processed_df.columns:
                    processed_df[column] = processed_df[column].astype(int)
            
            # Parse dates - critical for time series analysis
            if 'Date' in processed_df.columns:
                processed_df['Date'] = self._parse_dates(processed_df['Date'])
                # Add additional date columns for easy analysis
                processed_df['Year'] = processed_df['Date'].dt.year
                processed_df['Month'] = processed_df['Date'].dt.month
                processed_df['Week'] = processed_df['Date'].dt.isocalendar().week
            
            logger.info(f"Successfully processed sales data: {len(processed_df)} records")
            return processed_df
        
        except Exception as e:
            logger.error(f"Error processing sales data: {str(e)}")
            raise
    
    def _validate_columns(self, df: pd.DataFrame, required_columns: List[str]) -> None:
        """
        Validate if all required columns exist in the DataFrame.
        
        Args:
            df (pd.DataFrame): DataFrame to validate.
            required_columns (List[str]): List of required column names.
            
        Raises:
            ValueError: If any required columns are missing.
        """
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            error_msg = f"Missing required columns: {missing_columns}"
            logger.error(error_msg)
            raise ValueError(error_msg)
    
    def _parse_dates(self, date_series: pd.Series) -> pd.Series:
        """
        Parse date strings into datetime objects using multiple formats.
        
        Args:
            date_series (pd.Series): Series containing date strings.
            
        Returns:
            pd.Series: Series with parsed datetime objects.
        """
        # Try common date formats
        date_formats = [
            '%Y-%m-%d',          # 2023-01-31
            '%d/%m/%Y',          # 31/01/2023
            '%m/%d/%Y',          # 01/31/2023
            '%d-%m-%Y',          # 31-01-2023
            '%Y/%m/%d',          # 2023/01/31
            '%d.%m.%Y',          # 31.01.2023
            '%B %d, %Y',         # January 31, 2023
            '%d %B %Y',          # 31 January 2023
            '%Y-%m-%d %H:%M:%S'  # 2023-01-31 12:30:45
        ]
        
        result_series = pd.to_datetime(date_series, errors='coerce')
        
        # If standard parsing failed, try each format
        if result_series.isna().any():
            for fmt in date_formats:
                # For values that are still NaT, try this format
                mask = result_series.isna()
                if not mask.any():
                    break
                    
                result_series.loc[mask] = pd.to_datetime(
                    date_series.loc[mask], 
                    format=fmt, 
                    errors='coerce'
                )
        
        # Check if we still have unparsed dates
        if result_series.isna().any():
            logger.warning(f"Could not parse {result_series.isna().sum()} date values")
        
        return result_series
    
    def merge_inventory_and_sales(self, 
                                 inventory_df: pd.DataFrame, 
                                 sales_df: pd.DataFrame) -> pd.DataFrame:
        """
        Merge inventory and sales data based on SKU.
        
        Args:
            inventory_df (pd.DataFrame): Processed inventory data.
            sales_df (pd.DataFrame): Processed sales data.
            
        Returns:
            pd.DataFrame: Merged DataFrame with inventory and aggregated sales data.
        """
        if inventory_df.empty or sales_df.empty:
            logger.warning("Cannot merge: One or both DataFrames are empty")
            return pd.DataFrame()
        
        try:
            # Ensure both DataFrames have the required columns
            if 'SKU' not in inventory_df.columns:
                raise ValueError("Inventory DataFrame missing 'SKU' column")
                
            if 'Product Code (SKU)' not in sales_df.columns:
                raise ValueError("Sales DataFrame missing 'Product Code (SKU)' column")
            
            # Aggregate sales data by SKU
            sales_agg = sales_df.groupby('Product Code (SKU)').agg({
                'Units Sold': 'sum',
                'Total': 'sum',
                'Date': 'max'  # Last sale date
            }).reset_index()
            
            sales_agg.rename(columns={
                'Product Code (SKU)': 'SKU',
                'Units Sold': 'Total Units Sold',
                'Total': 'Total Sales Value',
                'Date': 'Last Sale Date'
            }, inplace=True)
            
            # Merge with inventory data
            merged_df = pd.merge(
                inventory_df,
                sales_agg,
                on='SKU',
                how='left'
            )
            
            # Handle NaN values for products with no sales
            merged_df['Total Units Sold'].fillna(0, inplace=True)
            merged_df['Total Sales Value'].fillna(0, inplace=True)
            
            logger.info(f"Successfully merged inventory and sales data: {len(merged_df)} records")
            return merged_df
            
        except Exception as e:
            logger.error(f"Error merging inventory and sales data: {str(e)}")
            raise