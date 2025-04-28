"""
Notification Module.

This module provides functionality to generate alert notifications
for inventory management based on stockout risks and replenishment needs.
"""

import logging
from typing import Dict, List, Union, Optional, Any, Tuple
from datetime import datetime

import pandas as pd

logger = logging.getLogger(__name__)


class NotificationGenerator:
    """
    A class for generating alert notifications for inventory management.
    
    This class creates structured, human-readable notifications based on
    stockout risks and replenishment recommendations.
    """
    
    def __init__(self):
        """Initialize the NotificationGenerator."""
        pass
    
    def generate_stockout_alerts(self, stockout_df: pd.DataFrame, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Generate alert notifications for items at risk of stockout.
        
        Args:
            stockout_df (pd.DataFrame): DataFrame with stockout risk information.
            limit (int, optional): Maximum number of alerts to generate. Defaults to 20.
                
        Returns:
            List[Dict[str, Any]]: List of alert notification dictionaries.
        """
        if stockout_df.empty:
            logger.warning("Empty stockout data provided for alert generation")
            return []
        
        try:
            # Filter for items with stockout risk
            at_risk = stockout_df[stockout_df['Stockout Risk'] != 'None'].copy()
            
            if at_risk.empty:
                logger.info("No items at risk of stockout found")
                return []
            
            # Sort by risk level and days until stockout
            risk_order = {'High': 0, 'Medium': 1, 'Low': 2}
            at_risk['Risk Priority'] = at_risk['Stockout Risk'].map(risk_order)
            at_risk = at_risk.sort_values(['Risk Priority', 'Days Until Stockout'])
            
            # Generate alerts for top items (limited by limit parameter)
            alerts = []
            for i, row in at_risk.head(limit).iterrows():
                severity = "CRITICAL" if row['Stockout Risk'] == 'High' else (
                    "WARNING" if row['Stockout Risk'] == 'Medium' else "INFO"
                )
                
                # Format days until stockout
                days = round(row['Days Until Stockout'])
                days_message = (
                    f"Projected to run out in {days} days" if days < 90
                    else "No immediate stockout risk"
                )
                
                # Create alert message
                message = (
                    f"{row['Product']} (SKU: {row['SKU']}) has {row['Stockout Risk']} stockout risk. "
                    f"Current stock: {row['Current Stock']} units. {days_message}. "
                    f"Forecasted demand: {row['Total Forecasted Demand']} units. "
                    f"Projected remaining: {max(0, round(row['Projected Remaining Stock']))} units."
                )
                
                alerts.append({
                    'timestamp': datetime.now().isoformat(),
                    'severity': severity,
                    'type': 'STOCKOUT_RISK',
                    'sku': row['SKU'],
                    'product': row['Product'],
                    'risk_level': row['Stockout Risk'],
                    'days_until_stockout': days,
                    'current_stock': row['Current Stock'],
                    'forecasted_demand': row['Total Forecasted Demand'],
                    'projected_remaining': max(0, round(row['Projected Remaining Stock'])),
                    'message': message
                })
            
            logger.info(f"Generated {len(alerts)} stockout alert notifications")
            return alerts
            
        except Exception as e:
            logger.error(f"Error generating stockout alerts: {str(e)}")
            raise
    
    def generate_replenishment_alerts(self, 
                                     recommendations_df: pd.DataFrame,
                                     limit: int = 20) -> List[Dict[str, Any]]:
        """
        Generate alert notifications for items requiring replenishment.
        
        Args:
            recommendations_df (pd.DataFrame): DataFrame with replenishment recommendations.
            limit (int, optional): Maximum number of alerts to generate. Defaults to 20.
                
        Returns:
            List[Dict[str, Any]]: List of alert notification dictionaries.
        """
        if recommendations_df.empty:
            logger.warning("Empty recommendations data provided for alert generation")
            return []
        
        try:
            # Filter for items with replenishment need
            needs_replenishment = recommendations_df[
                recommendations_df['Order Quantity'] > 0
            ].copy()
            
            if needs_replenishment.empty:
                logger.info("No items require replenishment")
                return []
            
            # Sort by priority
            priority_order = {'High': 0, 'Medium': 1, 'Low': 2, 'Low-Rotation': 3, 'None': 4}
            needs_replenishment['Priority Rank'] = needs_replenishment['Replenishment Priority'].map(priority_order)
            needs_replenishment = needs_replenishment.sort_values('Priority Rank')
            
            # Generate alerts for top items (limited by limit parameter)
            alerts = []
            for i, row in needs_replenishment.head(limit).iterrows():
                severity = "CRITICAL" if row['Replenishment Priority'] == 'High' else (
                    "WARNING" if row['Replenishment Priority'] == 'Medium' else "INFO"
                )
                
                # Create alert message
                message = (
                    f"{row['Product']} (SKU: {row['SKU']}) needs replenishment. "
                    f"Priority: {row['Replenishment Priority']}. "
                    f"Recommended action: {row['Recommended Action']}. "
                    f"Order quantity: {row['Order Quantity']} units. "
                    f"Current stock: {row['Current Stock']} units."
                )
                
                alerts.append({
                    'timestamp': datetime.now().isoformat(),
                    'severity': severity,
                    'type': 'REPLENISHMENT_NEEDED',
                    'sku': row['SKU'],
                    'product': row['Product'],
                    'priority': row['Replenishment Priority'],
                    'action': row['Recommended Action'],
                    'order_quantity': row['Order Quantity'],
                    'current_stock': row['Current Stock'],
                    'message': message
                })
            
            logger.info(f"Generated {len(alerts)} replenishment alert notifications")
            return alerts
            
        except Exception as e:
            logger.error(f"Error generating replenishment alerts: {str(e)}")
            raise
    
    def generate_low_rotation_alerts(self, 
                                    low_rotation_df: pd.DataFrame,
                                    limit: int = 20) -> List[Dict[str, Any]]:
        """
        Generate alert notifications for low rotation inventory items.
        
        Args:
            low_rotation_df (pd.DataFrame): DataFrame with low rotation item information.
            limit (int, optional): Maximum number of alerts to generate. Defaults to 20.
                
        Returns:
            List[Dict[str, Any]]: List of alert notification dictionaries.
        """
        if low_rotation_df.empty:
            logger.warning("Empty low rotation data provided for alert generation")
            return []
        
        try:
            # Sort by units sold (ascending)
            sorted_df = low_rotation_df.sort_values('Units Sold')
            
            # Generate alerts for top items (limited by limit parameter)
            alerts = []
            for i, row in sorted_df.head(limit).iterrows():
                product_name = row['Product Name'] if 'Product Name' in row else f"SKU {row['Product Code (SKU)']}"
                
                # Create alert message
                message = (
                    f"{product_name} (SKU: {row['Product Code (SKU)']}) is a low rotation item. "
                    f"Only {row['Units Sold']} units sold in {row['Analysis Period']}. "
                    f"Consider evaluating inventory strategy for this product."
                )
                
                alerts.append({
                    'timestamp': datetime.now().isoformat(),
                    'severity': "INFO",
                    'type': 'LOW_ROTATION',
                    'sku': row['Product Code (SKU)'],
                    'product': product_name,
                    'units_sold': row['Units Sold'],
                    'period': row['Analysis Period'],
                    'message': message
                })
            
            logger.info(f"Generated {len(alerts)} low rotation alert notifications")
            return alerts
            
        except Exception as e:
            logger.error(f"Error generating low rotation alerts: {str(e)}")
            raise
    
    def format_alerts_for_display(self, alerts: List[Dict[str, Any]]) -> str:
        """
        Format alerts into a human-readable text format.
        
        Args:
            alerts (List[Dict[str, Any]]): List of alert notification dictionaries.
                
        Returns:
            str: Formatted alerts as a string.
        """
        if not alerts:
            return "No alerts to display."
        
        formatted = []
        
        # Group alerts by type
        alerts_by_type = {}
        for alert in alerts:
            alert_type = alert['type']
            if alert_type not in alerts_by_type:
                alerts_by_type[alert_type] = []
            alerts_by_type[alert_type].append(alert)
        
        # Format each group
        for alert_type, alert_group in alerts_by_type.items():
            formatted.append(f"\n=== {alert_type.replace('_', ' ')} ALERTS ===")
            
            # Sort by severity
            severity_order = {"CRITICAL": 0, "WARNING": 1, "INFO": 2}
            sorted_alerts = sorted(alert_group, key=lambda x: severity_order.get(x['severity'], 3))
            
            for alert in sorted_alerts:
                severity_indicator = "🔴" if alert['severity'] == "CRITICAL" else (
                    "🟠" if alert['severity'] == "WARNING" else "🔵"
                )
                
                formatted.append(f"{severity_indicator} {alert['message']}")
        
        return "\n".join(formatted)