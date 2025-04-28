"""
Report Storage Module.

This module provides functionality to manage temporary storage
for generated reports and handle report expiration.
"""

import logging
import os
import shutil
from datetime import datetime, timedelta
from typing import Dict, List, Union, Optional, Any
import uuid

logger = logging.getLogger(__name__)


class ReportStorage:
    """
    A class for managing report storage and retrieval.
    
    This class handles temporary storage of generated reports,
    manages report expiration, and provides access to reports.
    """
    
    def __init__(self, 
                temp_dir: str = None,
                report_expiry_hours: int = 24):
        """
        Initialize the ReportStorage.
        
        Args:
            temp_dir (str, optional): Directory for temporary report storage.
                If not provided, will use TEMP_STORAGE_PATH env var or './data/temp'.
            report_expiry_hours (int, optional): Hours before reports expire. Defaults to 24.
        """
        if temp_dir is None:
            temp_dir = os.environ.get('TEMP_STORAGE_PATH', './data/temp')
            
        self.temp_dir = temp_dir
        self.report_expiry_hours = int(os.environ.get('REPORT_EXPIRY_HOURS', report_expiry_hours))
        
        # Create temp directory if it doesn't exist
        os.makedirs(self.temp_dir, exist_ok=True)
        
        logger.info(f"Report storage initialized. Path: {self.temp_dir}, Expiry: {self.report_expiry_hours} hours")
        
    def store_report(self, report_path: str, report_type: str = 'generic') -> Dict[str, Any]:
        """
        Store a report file and generate metadata.
        
        Args:
            report_path (str): Path to the report file.
            report_type (str, optional): Type of report. Defaults to 'generic'.
                
        Returns:
            Dict[str, Any]: Report metadata.
            
        Raises:
            FileNotFoundError: If the report file doesn't exist.
        """
        if not os.path.exists(report_path):
            error_msg = f"Report file not found: {report_path}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        
        try:
            # Generate a unique report ID
            report_id = str(uuid.uuid4())
            
            # Get filename from the path
            filename = os.path.basename(report_path)
            
            # If the report is not already in the temp directory, copy it there
            if os.path.dirname(report_path) != os.path.abspath(self.temp_dir):
                target_path = os.path.join(self.temp_dir, filename)
                shutil.copy2(report_path, target_path)
                stored_path = target_path
            else:
                stored_path = report_path
            
            # Get file stats
            file_stats = os.stat(stored_path)
            file_size = file_stats.st_size
            creation_time = datetime.fromtimestamp(file_stats.st_ctime)
            expiry_time = creation_time + timedelta(hours=self.report_expiry_hours)
            
            # Create metadata
            metadata = {
                'report_id': report_id,
                'filename': filename,
                'file_path': stored_path,
                'file_size': file_size,
                'report_type': report_type,
                'creation_time': creation_time.isoformat(),
                'expiry_time': expiry_time.isoformat(),
                'status': 'active'
            }
            
            logger.info(f"Report stored successfully. ID: {report_id}, Path: {stored_path}")
            return metadata
            
        except Exception as e:
            logger.error(f"Error storing report: {str(e)}")
            raise
    
    def get_report_path(self, report_id_or_filename: str) -> Optional[str]:
        """
        Get the path to a stored report.
        
        Args:
            report_id_or_filename (str): Report ID or filename.
                
        Returns:
            Optional[str]: Path to the report file, or None if not found.
        """
        try:
            # Check if the input is a full filename
            filename = report_id_or_filename
            file_path = os.path.join(self.temp_dir, filename)
            
            if os.path.exists(file_path):
                return file_path
            
            # If not found, treat input as a report ID prefix and look for matching files
            for file in os.listdir(self.temp_dir):
                if file.endswith(report_id_or_filename) or file.startswith(report_id_or_filename):
                    return os.path.join(self.temp_dir, file)
            
            logger.warning(f"Report not found: {report_id_or_filename}")
            return None
            
        except Exception as e:
            logger.error(f"Error retrieving report: {str(e)}")
            return None
    
    def list_reports(self, report_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List all available reports.
        
        Args:
            report_type (str, optional): Filter by report type. Defaults to None.
                
        Returns:
            List[Dict[str, Any]]: List of report metadata.
        """
        try:
            reports = []
            
            for filename in os.listdir(self.temp_dir):
                if filename.endswith('.xlsx'):  # Only list Excel reports
                    file_path = os.path.join(self.temp_dir, filename)
                    file_stats = os.stat(file_path)
                    
                    creation_time = datetime.fromtimestamp(file_stats.st_ctime)
                    expiry_time = creation_time + timedelta(hours=self.report_expiry_hours)
                    
                    # Determine report type from filename
                    if 'inventory' in filename.lower():
                        detected_type = 'inventory'
                    elif 'sales' in filename.lower():
                        detected_type = 'sales'
                    elif 'forecast' in filename.lower():
                        detected_type = 'forecast'
                    else:
                        detected_type = 'generic'
                    
                    # Skip if filtering by type and this doesn't match
                    if report_type and detected_type != report_type:
                        continue
                    
                    # Check if report is expired
                    is_expired = datetime.now() > expiry_time
                    
                    report_info = {
                        'filename': filename,
                        'file_path': file_path,
                        'file_size': file_stats.st_size,
                        'report_type': detected_type,
                        'creation_time': creation_time.isoformat(),
                        'expiry_time': expiry_time.isoformat(),
                        'status': 'expired' if is_expired else 'active'
                    }
                    
                    reports.append(report_info)
            
            # Sort by creation time (newest first)
            reports.sort(key=lambda x: x['creation_time'], reverse=True)
            
            return reports
            
        except Exception as e:
            logger.error(f"Error listing reports: {str(e)}")
            return []
    
    def cleanup_expired_reports(self) -> int:
        """
        Remove expired reports from storage.
                
        Returns:
            int: Number of reports removed.
        """
        try:
            removed_count = 0
            now = datetime.now()
            
            for filename in os.listdir(self.temp_dir):
                if not filename.endswith('.xlsx'):
                    continue
                
                file_path = os.path.join(self.temp_dir, filename)
                file_stats = os.stat(file_path)
                
                creation_time = datetime.fromtimestamp(file_stats.st_ctime)
                expiry_time = creation_time + timedelta(hours=self.report_expiry_hours)
                
                if now > expiry_time:
                    os.remove(file_path)
                    removed_count += 1
                    logger.info(f"Removed expired report: {filename}")
            
            logger.info(f"Cleanup completed. Removed {removed_count} expired reports.")
            return removed_count
            
        except Exception as e:
            logger.error(f"Error cleaning up expired reports: {str(e)}")
            return 0
    
    def delete_report(self, report_id_or_filename: str) -> bool:
        """
        Delete a specific report.
        
        Args:
            report_id_or_filename (str): Report ID or filename.
                
        Returns:
            bool: True if deleted successfully, False otherwise.
        """
        try:
            file_path = self.get_report_path(report_id_or_filename)
            
            if file_path and os.path.exists(file_path):
                os.remove(file_path)
                logger.info(f"Report deleted: {file_path}")
                return True
            
            logger.warning(f"Could not delete report, not found: {report_id_or_filename}")
            return False
            
        except Exception as e:
            logger.error(f"Error deleting report: {str(e)}")
            return False