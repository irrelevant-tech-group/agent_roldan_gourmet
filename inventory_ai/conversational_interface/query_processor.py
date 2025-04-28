"""
Query Processor Module.

This module provides functionality to understand and process
natural language queries related to inventory and sales data.
"""

import logging
import re
from typing import Dict, List, Union, Optional, Any, Tuple
from datetime import datetime, timedelta

import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Ensure required NLTK datasets are available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

logger = logging.getLogger(__name__)


class QueryProcessor:
    """
    A class for processing natural language inventory and sales queries.
    
    This class analyzes user queries to determine intent and extract
    parameters for generating appropriate responses.
    """
    
    # Define query types and their keywords
    QUERY_TYPES = {
        'inventory_status': [
            'inventory', 'stock', 'available', 'current', 'on hand', 'how many', 
            'do we have', 'in stock', 'remaining'
        ],
        'stockout_risk': [
            'stockout', 'risk', 'running out', 'low stock', 'need to order',
            'reorder', 'replenish', 'about to run out', 'out of stock'
        ],
        'sales_report': [
            'sales', 'sold', 'revenue', 'income', 'profit', 'selling', 
            'performance', 'how much', 'how many sold', 'revenue', 'report'
        ],
        'product_info': [
            'product', 'item', 'sku', 'description', 'details', 'tell me about',
            'information about', 'specs', 'specification'
        ],
        'forecast': [
            'forecast', 'predict', 'projection', 'future sales', 'expected',
            'anticipated', 'demand', 'trending', 'estimate'
        ],
        'low_rotation': [
            'low rotation', 'slow moving', 'not selling', 'obsolete', 'dead stock',
            'slow seller', 'poor performance', 'inactive'
        ]
    }
    
    # Time frame related keywords
    TIME_FRAMES = {
        'today': {'days': 0},
        'yesterday': {'days': 1},
        'this week': {'weeks': 0},
        'last week': {'weeks': 1},
        'this month': {'months': 0},
        'last month': {'months': 1},
        'this year': {'years': 0},
        'last year': {'years': 1},
        'past week': {'weeks': 1},
        'past month': {'months': 1},
        'past 30 days': {'days': 30},
        'past 60 days': {'days': 60},
        'past 90 days': {'days': 90},
        'past quarter': {'months': 3},
        'past 6 months': {'months': 6},
        'past year': {'years': 1}
    }
    
    # Common month names for date extraction
    MONTHS = {
        'january': 1, 'february': 2, 'march': 3, 'april': 4, 'may': 5, 'june': 6,
        'july': 7, 'august': 8, 'september': 9, 'october': 10, 'november': 11, 'december': 12,
        'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'jun': 6, 'jul': 7, 'aug': 8, 'sep': 9, 
        'sept': 9, 'oct': 10, 'nov': 11, 'dec': 12
    }
    
    def __init__(self):
        """Initialize the QueryProcessor."""
        self.stop_words = set(stopwords.words('english'))
    
    def process_query(self, query: str) -> Dict[str, Any]:
        """
        Process a natural language query and determine its intent and parameters.
        
        Args:
            query (str): The natural language query from the user.
                
        Returns:
            Dict[str, Any]: Query intent and extracted parameters.
        """
        if not query or not isinstance(query, str):
            return {'intent': 'unknown', 'confidence': 0, 'parameters': {}}
        
        # Normalize the query
        normalized_query = self._normalize_query(query)
        
        # Determine the query type
        query_type, confidence = self._determine_query_type(normalized_query)
        
        # Extract parameters
        parameters = self._extract_parameters(normalized_query, query_type)
        
        # Create structured result
        result = {
            'intent': query_type,
            'confidence': confidence,
            'parameters': parameters,
            'original_query': query,
            'normalized_query': normalized_query,
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"Processed query: {query_type} (confidence: {confidence:.2f})")
        return result
    
    def _normalize_query(self, query: str) -> str:
        """
        Normalize a query by converting to lowercase and removing punctuation.
        
        Args:
            query (str): The original query.
                
        Returns:
            str: Normalized query.
        """
        # Convert to lowercase
        normalized = query.lower()
        
        # Remove punctuation except for dates and percentages
        normalized = re.sub(r'[^\w\s/%\-.]', ' ', normalized)
        
        # Standardize whitespace
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        
        return normalized
    
    def _determine_query_type(self, query: str) -> Tuple[str, float]:
        """
        Determine the type of query based on keyword matching.
        
        Args:
            query (str): The normalized query.
                
        Returns:
            Tuple[str, float]: Query type and confidence score.
        """
        # Check for each query type
        scores = {}
        
        for query_type, keywords in self.QUERY_TYPES.items():
            score = 0
            max_score = len(keywords)
            
            for keyword in keywords:
                if keyword in query:
                    score += 1
            
            if max_score > 0:
                confidence = score / max_score
            else:
                confidence = 0
                
            scores[query_type] = confidence
        
        # Find the query type with the highest score
        best_type = max(scores.items(), key=lambda x: x[1])
        
        # If confidence is too low, mark as unknown
        if best_type[1] < 0.2:
            return 'unknown', 0.0
        
        return best_type
    
    def _extract_parameters(self, query: str, query_type: str) -> Dict[str, Any]:
        """
        Extract relevant parameters from the query.
        
        Args:
            query (str): The normalized query.
            query_type (str): The determined query type.
                
        Returns:
            Dict[str, Any]: Extracted parameters.
        """
        parameters = {}
        
        # Extract product/SKU information
        sku = self._extract_sku(query)
        if sku:
            parameters['sku'] = sku
        
        product_name = self._extract_product_name(query)
        if product_name:
            parameters['product_name'] = product_name
        
        # Extract time frame information
        time_frame = self._extract_time_frame(query)
        if time_frame:
            parameters['time_frame'] = time_frame
        
        # Extract date range
        date_range = self._extract_date_range(query)
        if date_range:
            parameters['date_range'] = date_range
        
        # Extract quantity or limit
        quantity = self._extract_quantity(query)
        if quantity:
            parameters['quantity'] = quantity
        
        # Additional parameters based on query type
        if query_type == 'stockout_risk':
            parameters['risk_level'] = self._extract_risk_level(query)
        
        if query_type == 'sales_report':
            parameters['report_type'] = self._extract_report_type(query)
        
        return parameters
    
    def _extract_sku(self, query: str) -> Optional[str]:
        """
        Extract SKU from the query.
        
        Args:
            query (str): The normalized query.
                
        Returns:
            Optional[str]: Extracted SKU or None.
        """
        # Look for SKU patterns (alphanumeric codes)
        sku_patterns = [
            r'\bsku\s*[:#-]?\s*([a-z0-9-]+)',  # SKU: ABC123
            r'\bproduct\s*[:#-]?\s*([a-z0-9-]+)',  # Product: ABC123
            r'\bitem\s*[:#-]?\s*([a-z0-9-]+)',  # Item: ABC123
            r'\b([a-z][a-z0-9]{2,10})\b'  # Standalone SKU like ABC123
        ]
        
        for pattern in sku_patterns:
            match = re.search(pattern, query)
            if match:
                return match.group(1).upper()
        
        return None
    
    def _extract_product_name(self, query: str) -> Optional[str]:
        """
        Extract product name from the query.
        
        Args:
            query (str): The normalized query.
                
        Returns:
            Optional[str]: Extracted product name or None.
        """
        # Look for product name patterns
        product_patterns = [
            r'product\s+name\s*[:#]?\s*([a-z0-9 ]+)',  # Product name: Widget
            r'for\s+product\s+([a-z0-9 ]+)',  # For product Widget
            r'for\s+([a-z0-9 ]{3,30})\b'  # For Widget
        ]
        
        for pattern in product_patterns:
            match = re.search(pattern, query)
            if match:
                product_name = match.group(1).strip()
                # Remove common stop words from the beginning
                for stop_word in ['the', 'a', 'an', 'product', 'item']:
                    if product_name.startswith(stop_word + ' '):
                        product_name = product_name[len(stop_word) + 1:]
                return product_name.title()
        
        return None
    
    def _extract_time_frame(self, query: str) -> Optional[Dict[str, Any]]:
        """
        Extract time frame information from the query.
        
        Args:
            query (str): The normalized query.
                
        Returns:
            Optional[Dict[str, Any]]: Time frame information or None.
        """
        # Check for predefined time frames
        for time_frame, offset in self.TIME_FRAMES.items():
            if time_frame in query:
                # Calculate dates based on the time frame
                end_date = datetime.now()
                
                if 'days' in offset:
                    if offset['days'] > 0:
                        start_date = end_date - timedelta(days=offset['days'])
                    else:
                        # For "today"
                        start_date = end_date.replace(hour=0, minute=0, second=0, microsecond=0)
                
                elif 'weeks' in offset:
                    if offset['weeks'] > 0:
                        start_date = end_date - timedelta(weeks=offset['weeks'])
                    else:
                        # For "this week" - start from Monday
                        weekday = end_date.weekday()  # 0 is Monday, 6 is Sunday
                        start_date = end_date - timedelta(days=weekday)
                        start_date = start_date.replace(hour=0, minute=0, second=0, microsecond=0)
                
                elif 'months' in offset:
                    if offset['months'] > 0:
                        # Go back N months
                        new_month = end_date.month - offset['months']
                        new_year = end_date.year
                        while new_month <= 0:
                            new_month += 12
                            new_year -= 1
                        start_date = end_date.replace(year=new_year, month=new_month, day=1, 
                                                   hour=0, minute=0, second=0, microsecond=0)
                    else:
                        # Start of current month
                        start_date = end_date.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
                
                elif 'years' in offset:
                    if offset['years'] > 0:
                        start_date = end_date.replace(year=end_date.year - offset['years'], 
                                                   month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
                    else:
                        # Start of current year
                        start_date = end_date.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
                
                else:
                    # Default: past 30 days
                    start_date = end_date - timedelta(days=30)
                
                return {
                    'name': time_frame,
                    'start_date': start_date.isoformat(),
                    'end_date': end_date.isoformat()
                }
        
        return None
    
    def _extract_date_range(self, query: str) -> Optional[Dict[str, str]]:
        """
        Extract specific date range from the query.
        
        Args:
            query (str): The normalized query.
                
        Returns:
            Optional[Dict[str, str]]: Date range information or None.
        """
        # Look for specific dates in various formats
        date_patterns = [
            # MM/DD/YYYY or MM-DD-YYYY
            r'(\d{1,2})[/-](\d{1,2})[/-](20\d{2})',
            
            # Month name and year
            r'(january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec)\s+(20\d{2})',
            
            # Year only
            r'\b(20\d{2})\b'
        ]
        
        extracted_dates = []
        
        for pattern in date_patterns:
            matches = re.finditer(pattern, query)
            for match in matches:
                if len(match.groups()) == 3:  # MM/DD/YYYY
                    month, day, year = match.groups()
                    try:
                        date_obj = datetime(int(year), int(month), int(day))
                        extracted_dates.append(date_obj)
                    except ValueError:
                        continue
                
                elif len(match.groups()) == 2:
                    if match.group(1) in self.MONTHS:  # Month name and year
                        month_name, year = match.groups()
                        month_num = self.MONTHS[month_name.lower()]
                        try:
                            date_obj = datetime(int(year), month_num, 1)
                            extracted_dates.append(date_obj)
                        except ValueError:
                            continue
                    else:  # Year only, assume Jan 1
                        year = match.group(1)
                        try:
                            date_obj = datetime(int(year), 1, 1)
                            extracted_dates.append(date_obj)
                        except ValueError:
                            continue
        
        # If we found any dates
        if extracted_dates:
            # Sort dates
            extracted_dates.sort()
            
            # If only one date is found, assume it's the start date and end date is now
            if len(extracted_dates) == 1:
                start_date = extracted_dates[0]
                
                # If a month was specified, set the end date to the end of that month
                if "month" in query or any(month in query for month in self.MONTHS.keys()):
                    # Set to the last day of the month
                    if start_date.month == 12:
                        end_date = datetime(start_date.year + 1, 1, 1) - timedelta(days=1)
                    else:
                        end_date = datetime(start_date.year, start_date.month + 1, 1) - timedelta(days=1)
                
                # If a year was specified, set the end date to the end of that year
                elif "year" in query:
                    end_date = datetime(start_date.year, 12, 31)
                
                # Default: assume date is a specific day
                else:
                    end_date = start_date.replace(hour=23, minute=59, second=59)
            
            else:
                # Use the first and last date
                start_date = extracted_dates[0]
                end_date = extracted_dates[-1]
                
                # If end date is just a date without time, make it end of day
                if end_date.hour == 0 and end_date.minute == 0:
                    end_date = end_date.replace(hour=23, minute=59, second=59)
            
            return {
                'start_date': start_date.isoformat(),
                'end_date': end_date.isoformat()
            }
        
        # Check for "between" date range
        between_pattern = r'between\s+(.*?)\s+and\s+(.*?)(\s|$)'
        match = re.search(between_pattern, query)
        if match:
            start_text, end_text = match.groups()[:2]
            
            # Try to parse these text fragments
            date_range = self._parse_date_text(start_text, end_text)
            if date_range:
                return date_range
        
        return None
    
    def _parse_date_text(self, start_text: str, end_text: str) -> Optional[Dict[str, str]]:
        """
        Parse date text fragments into a date range.
        
        Args:
            start_text (str): Text describing the start date.
            end_text (str): Text describing the end date.
                
        Returns:
            Optional[Dict[str, str]]: Date range information or None.
        """
        start_date = None
        end_date = None
        
        # Check for month names
        for text, date_var in [(start_text, 'start_date'), (end_text, 'end_date')]:
            for month_name, month_num in self.MONTHS.items():
                if month_name in text:
                    # Try to extract year
                    year_match = re.search(r'(20\d{2})', text)
                    year = int(year_match.group(1)) if year_match else datetime.now().year
                    
                    # Create date object
                    try:
                        if date_var == 'start_date':
                            start_date = datetime(year, month_num, 1)
                        else:
                            # Last day of the month
                            if month_num == 12:
                                end_date = datetime(year + 1, 1, 1) - timedelta(days=1)
                            else:
                                end_date = datetime(year, month_num + 1, 1) - timedelta(days=1)
                            end_date = end_date.replace(hour=23, minute=59, second=59)
                    except ValueError:
                        continue
        
        if start_date and end_date:
            return {
                'start_date': start_date.isoformat(),
                'end_date': end_date.isoformat()
            }
        
        return None
    
    def _extract_quantity(self, query: str) -> Optional[int]:
        """
        Extract quantity or limit from the query.
        
        Args:
            query (str): The normalized query.
                
        Returns:
            Optional[int]: Extracted quantity or None.
        """
        # Look for quantity patterns
        quantity_patterns = [
            r'top\s+(\d+)',  # top 10
            r'(\d+)\s+most',  # 10 most
            r'limit\s+(\d+)',  # limit 10
            r'(\d+)\s+items',  # 10 items
            r'show\s+(\d+)',  # show 10
            r'list\s+(\d+)'   # list 10
        ]
        
        for pattern in quantity_patterns:
            match = re.search(pattern, query)
            if match:
                try:
                    return int(match.group(1))
                except ValueError:
                    continue
        
        return None
    
    def _extract_risk_level(self, query: str) -> Optional[str]:
        """
        Extract risk level from a stockout risk query.
        
        Args:
            query (str): The normalized query.
                
        Returns:
            Optional[str]: Risk level or None.
        """
        if 'high risk' in query or 'critical' in query:
            return 'high'
        elif 'medium risk' in query or 'moderate' in query:
            return 'medium'
        elif 'low risk' in query or 'minimal' in query:
            return 'low'
        
        return None
    
    def _extract_report_type(self, query: str) -> Optional[str]:
        """
        Extract report type from a report query.
        
        Args:
            query (str): The normalized query.
                
        Returns:
            Optional[str]: Report type or None.
        """
        if 'summary' in query:
            return 'summary'
        elif 'detail' in query or 'detailed' in query:
            return 'detailed'
        elif 'excel' in query or 'download' in query or 'export' in query:
            return 'excel'
        
        return 'summary'  # Default to summary