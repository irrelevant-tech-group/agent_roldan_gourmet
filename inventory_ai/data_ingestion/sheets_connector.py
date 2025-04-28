"""
Google Sheets Connector Module.

This module provides functionality to securely connect to Google Sheets
using Service Account authentication and retrieve data from inventory and sales sheets.
"""

import os
import logging
from typing import Dict, List, Optional, Any, Union

import gspread
from oauth2client.service_account import ServiceAccountCredentials
import pandas as pd

logger = logging.getLogger(__name__)


class SheetsConnector:
    """
    A connector class for securely accessing Google Sheets data.
    
    This class handles authentication with Google Sheets API using service account
    credentials and provides methods to fetch and process data from inventory
    and sales sheets.
    """
    
    # Define the scope required for Google Sheets API
    SCOPES = [
        'https://spreadsheets.google.com/feeds',
        'https://www.googleapis.com/auth/drive'
    ]
    
    def __init__(self, credentials_file: str = None):
        """
        Initialize the SheetsConnector with Google API credentials.
        
        Args:
            credentials_file (str, optional): Path to the Google API credentials JSON file.
                If not provided, will look for path in GOOGLE_SHEETS_CREDENTIALS_FILE env var.
        
        Raises:
            FileNotFoundError: If the credentials file doesn't exist.
            ValueError: If credentials cannot be loaded properly.
        """
        if credentials_file is None:
            credentials_file = os.environ.get('GOOGLE_SHEETS_CREDENTIALS_FILE')
            
        if not credentials_file or not os.path.exists(credentials_file):
            raise FileNotFoundError(
                f"Google API credentials file not found: {credentials_file}"
            )
        
        try:
            # Authenticate with Google Sheets API
            credentials = ServiceAccountCredentials.from_json_keyfile_name(
                credentials_file, self.SCOPES
            )
            self.client = gspread.authorize(credentials)
            logger.info("Successfully authenticated with Google Sheets API")
        except Exception as e:
            logger.error(f"Failed to authenticate with Google Sheets API: {str(e)}")
            raise ValueError(f"Failed to authenticate with Google Sheets: {str(e)}")
    
    def get_sheet(self, sheet_id: str, worksheet_index: int = 0) -> gspread.Worksheet:
        """
        Get a specific worksheet from a Google Sheet.
        
        Args:
            sheet_id (str): The ID of the Google Sheet.
            worksheet_index (int, optional): Index of the worksheet to access. Defaults to 0.
        
        Returns:
            gspread.Worksheet: The requested worksheet object.
            
        Raises:
            Exception: If the sheet or worksheet cannot be accessed.
        """
        try:
            # Open the spreadsheet and get the specified worksheet
            spreadsheet = self.client.open_by_key(sheet_id)
            worksheet = spreadsheet.get_worksheet(worksheet_index)
            return worksheet
        except Exception as e:
            logger.error(f"Failed to access sheet {sheet_id}: {str(e)}")
            raise
    
    def get_sheet_data(self, 
                       sheet_id: str, 
                       worksheet_index: int = 0,
                       as_dataframe: bool = True) -> Union[List[Dict], pd.DataFrame]:
        """
        Get all data from a specific worksheet as a list of dictionaries or DataFrame.
        
        Args:
            sheet_id (str): The ID of the Google Sheet.
            worksheet_index (int, optional): Index of the worksheet. Defaults to 0.
            as_dataframe (bool, optional): Whether to return data as pandas DataFrame.
                Defaults to True.
        
        Returns:
            Union[List[Dict], pd.DataFrame]: Sheet data as list of dicts or DataFrame.
            
        Raises:
            Exception: If data cannot be retrieved from the sheet.
        """
        try:
            worksheet = self.get_sheet(sheet_id, worksheet_index)
            # Get all records as dictionaries
            data = worksheet.get_all_records()
            
            if not data:
                logger.warning(f"No data found in sheet {sheet_id}, worksheet {worksheet_index}")
            
            if as_dataframe:
                return pd.DataFrame(data)
            return data
        except Exception as e:
            logger.error(f"Failed to get data from sheet {sheet_id}: {str(e)}")
            raise
    
    def get_inventory_data(self, inventory_sheet_id: Optional[str] = None) -> pd.DataFrame:
        """
        Get inventory data from the inventory Google Sheet.
        
        Args:
            inventory_sheet_id (str, optional): The ID of the inventory Google Sheet.
                If not provided, will look for ID in INVENTORY_SHEET_ID env var.
        
        Returns:
            pd.DataFrame: Dataframe containing inventory data.
            
        Raises:
            ValueError: If inventory sheet ID is not provided or found.
        """
        if inventory_sheet_id is None:
            inventory_sheet_id = os.environ.get('INVENTORY_SHEET_ID')
            
        if not inventory_sheet_id:
            raise ValueError("Inventory Sheet ID not provided")
        
        logger.info(f"Fetching inventory data from sheet {inventory_sheet_id}")
        return self.get_sheet_data(inventory_sheet_id)
    
    def get_sales_data(self, sales_sheet_id: Optional[str] = None) -> pd.DataFrame:
        """
        Get sales data from the sales Google Sheet.
        
        Args:
            sales_sheet_id (str, optional): The ID of the sales Google Sheet.
                If not provided, will look for ID in SALES_SHEET_ID env var.
        
        Returns:
            pd.DataFrame: Dataframe containing sales data.
            
        Raises:
            ValueError: If sales sheet ID is not provided or found.
        """
        if sales_sheet_id is None:
            sales_sheet_id = os.environ.get('SALES_SHEET_ID')
            
        if not sales_sheet_id:
            raise ValueError("Sales Sheet ID not provided")
        
        logger.info(f"Fetching sales data from sheet {sales_sheet_id}")
        return self.get_sheet_data(sales_sheet_id)
    
    def update_sheet_values(self, 
                            sheet_id: str, 
                            worksheet_index: int,
                            range_name: str, 
                            values: List[List[Any]]) -> Dict:
        """
        Update values in a specific range of a worksheet.
        
        Args:
            sheet_id (str): The ID of the Google Sheet.
            worksheet_index (int): Index of the worksheet to update.
            range_name (str): The A1 notation of the range to update.
            values (List[List[Any]]): The values to update in the range.
        
        Returns:
            Dict: Response from the API containing the update result.
            
        Raises:
            Exception: If the update operation fails.
        """
        try:
            worksheet = self.get_sheet(sheet_id, worksheet_index)
            # Update the range with provided values
            result = worksheet.update(range_name, values)
            logger.info(f"Updated range {range_name} in sheet {sheet_id}")
            return result
        except Exception as e:
            logger.error(f"Failed to update range {range_name} in sheet {sheet_id}: {str(e)}")
            raise