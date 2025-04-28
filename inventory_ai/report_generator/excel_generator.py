"""
Excel Generator Module

This module provides the `ExcelReportGenerator` class, which can create
well-formatted Excel workbooks for inventory, sales and forecast analysis.
Each report comes with summary dashboards and visual charts generated
with XlsxWriter.

Dependencies
------------
pandas, numpy, xlsxwriter
"""

from __future__ import annotations

import logging
import os
import uuid
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
import xlsxwriter
from xlsxwriter.utility import xl_rowcol_to_cell

logger = logging.getLogger(__name__)


class ExcelReportGenerator:
    """
    Generate structured Excel reports (inventory, sales, forecast) with
    conditional formatting and charts.
    """

    def __init__(self, temp_dir: str | None = None) -> None:
        """
        Parameters
        ----------
        temp_dir : str, optional
            Where to save the generated files.  
            Falls back to the `TEMP_STORAGE_PATH` environment variable or
            `./data/temp` if nothing is provided.
        """
        self.temp_dir = (
            temp_dir if temp_dir is not None
            else os.environ.get("TEMP_STORAGE_PATH", "./data/temp")
        )
        os.makedirs(self.temp_dir, exist_ok=True)

    # --------------------------------------------------------------------- #
    #  INVENTORY REPORT                                                     #
    # --------------------------------------------------------------------- #

    def generate_inventory_report(
        self,
        inventory_df: pd.DataFrame,
        stockout_df: pd.DataFrame | None = None,
        replenishment_df: pd.DataFrame | None = None,
    ) -> str:
        """
        Build an Excel file with an inventory summary, stock-out risks,
        replenishment recommendations and an analysis dashboard.

        Returns
        -------
        str
            Path to the created workbook.
        """
        if inventory_df.empty:
            raise ValueError("Cannot generate report: Inventory data is empty")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"inventory_report_{timestamp}_{uuid.uuid4().hex[:8]}.xlsx"
        filepath = os.path.join(self.temp_dir, filename)

        try:
            with pd.ExcelWriter(filepath, engine="xlsxwriter") as writer:
                wb = writer.book

                # ------------------------------------------------------------------
                # Common formats
                # ------------------------------------------------------------------
                header_fmt = wb.add_format(
                    {
                        "bold": True,
                        "text_wrap": True,
                        "valign": "top",
                        "fg_color": "#D7E4BC",
                        "border": 1,
                    }
                )
                low_fmt = wb.add_format(
                    {"bg_color": "#FFC7CE", "font_color": "#9C0006"}
                )
                medium_fmt = wb.add_format(
                    {"bg_color": "#FFEB9C", "font_color": "#9C5700"}
                )
                good_fmt = wb.add_format(
                    {"bg_color": "#C6EFCE", "font_color": "#006100"}
                )
                currency_fmt = wb.add_format({"num_format": "$#,##0.00"})

                # ------------------------------------------------------------------
                # Inventory summary
                # ------------------------------------------------------------------
                inventory_df.to_excel(writer, sheet_name="Inventory Summary", index=False)
                ws = writer.sheets["Inventory Summary"]

                for col, name in enumerate(inventory_df.columns):
                    ws.write(0, col, name, header_fmt)

                ws.set_column("A:A", 12)
                ws.set_column("B:B", 30)
                ws.set_column("C:Z", 15)

                if {"Current Stock", "Minimum Stock"} <= set(inventory_df.columns):
                    cur_col = inventory_df.columns.get_loc("Current Stock")
                    min_col = inventory_df.columns.get_loc("Minimum Stock")
                    last = len(inventory_df)

                    # <= minimum  ➜ low
                    ws.conditional_format(
                        1,
                        cur_col,
                        last,
                        cur_col,
                        {
                            "type": "formula",
                            "criteria": f"=${xl_rowcol_to_cell(1, cur_col, False, True)}<="
                            f"${xl_rowcol_to_cell(1, min_col, False, True)}",
                            "format": low_fmt,
                        },
                    )
                    # between 1× and 1.5× minimum ➜ medium
                    ws.conditional_format(
                        1,
                        cur_col,
                        last,
                        cur_col,
                        {
                            "type": "formula",
                            "criteria": f"AND("
                            f"${xl_rowcol_to_cell(1, cur_col, False, True)}>"
                            f"${xl_rowcol_to_cell(1, min_col, False, True)},"
                            f"${xl_rowcol_to_cell(1, cur_col, False, True)}<="
                            f"${xl_rowcol_to_cell(1, min_col, False, True)}*1.5)",
                            "format": medium_fmt,
                        },
                    )
                    # >1.5× minimum ➜ good
                    ws.conditional_format(
                        1,
                        cur_col,
                        last,
                        cur_col,
                        {
                            "type": "formula",
                            "criteria": f"=${xl_rowcol_to_cell(1, cur_col, False, True)}>"
                            f"${xl_rowcol_to_cell(1, min_col, False, True)}*1.5",
                            "format": good_fmt,
                        },
                    )

                # ------------------------------------------------------------------
                # Stock-out risk sheet
                # ------------------------------------------------------------------
                if stockout_df is not None and not stockout_df.empty:
                    stockout_df.to_excel(writer, sheet_name="Stockout Risks", index=False)
                    ws = writer.sheets["Stockout Risks"]

                    for col, name in enumerate(stockout_df.columns):
                        ws.write(0, col, name, header_fmt)

                    ws.set_column("A:A", 12)
                    ws.set_column("B:B", 30)
                    ws.set_column("C:Z", 15)

                    if "Stockout Risk" in stockout_df.columns:
                        risk_col = stockout_df.columns.get_loc("Stockout Risk")
                        last = len(stockout_df)
                        for risk, fmt in (
                            ("High", low_fmt),
                            ("Medium", medium_fmt),
                            ("Low", good_fmt),
                        ):
                            ws.conditional_format(
                                1,
                                risk_col,
                                last,
                                risk_col,
                                {
                                    "type": "text",
                                    "criteria": "containing",
                                    "value": risk,
                                    "format": fmt,
                                },
                            )

                # ------------------------------------------------------------------
                # Replenishment recommendations
                # ------------------------------------------------------------------
                if replenishment_df is not None and not replenishment_df.empty:
                    replenishment_df.to_excel(
                        writer, sheet_name="Replenishment", index=False
                    )
                    ws = writer.sheets["Replenishment"]

                    for col, name in enumerate(replenishment_df.columns):
                        ws.write(0, col, name, header_fmt)

                    ws.set_column("A:A", 12)
                    ws.set_column("B:B", 30)
                    ws.set_column("C:Z", 15)

                    if "Replenishment Priority" in replenishment_df.columns:
                        pr_col = replenishment_df.columns.get_loc(
                            "Replenishment Priority"
                        )
                        last = len(replenishment_df)
                        for prio, fmt in (
                            ("High", low_fmt),
                            ("Medium", medium_fmt),
                            ("Low", good_fmt),
                        ):
                            ws.conditional_format(
                                1,
                                pr_col,
                                last,
                                pr_col,
                                {
                                    "type": "text",
                                    "criteria": "containing",
                                    "value": prio,
                                    "format": fmt,
                                },
                            )

                # ------------------------------------------------------------------
                # Dashboard
                # ------------------------------------------------------------------
                self._add_inventory_dashboard(wb, inventory_df, stockout_df)

            logger.info("Inventory report generated: %s", filepath)
            return filepath

        except Exception as exc:
            logger.exception("Error generating inventory report: %s", exc)
            raise

    # --------------------------------------------------------------------- #
    #  SALES REPORT                                                         #
    # --------------------------------------------------------------------- #

    def generate_sales_report(
        self,
        sales_df: pd.DataFrame,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> str:
        """
        Build an Excel file with detailed sales, summaries and trends.

        Returns
        -------
        str
            Path to the created workbook.
        """
        if sales_df.empty:
            raise ValueError("Cannot generate report: Sales data is empty")

        required = {"Product Code (SKU)", "Date", "Units Sold"}
        missing = required - set(sales_df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")

        if not pd.api.types.is_datetime64_any_dtype(sales_df["Date"]):
            sales_df["Date"] = pd.to_datetime(sales_df["Date"], errors="coerce")

        df = sales_df.copy()
        if start_date is not None:
            df = df[df["Date"] >= start_date]
        if end_date is not None:
            df = df[df["Date"] <= end_date]
        if df.empty:
            raise ValueError("No sales within the specified date range")

        date_tag = ""
        if start_date and end_date:
            date_tag = f"{start_date:%Y%m%d}_to_{end_date:%Y%m%d}"
        elif start_date:
            date_tag = f"from_{start_date:%Y%m%d}"
        elif end_date:
            date_tag = f"until_{end_date:%Y%m%d}"

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"sales_report_{date_tag}_{timestamp}.xlsx"
        filepath = os.path.join(self.temp_dir, filename)

        try:
            with pd.ExcelWriter(filepath, engine="xlsxwriter") as writer:
                wb = writer.book

                header_fmt = wb.add_format(
                    {
                        "bold": True,
                        "text_wrap": True,
                        "valign": "top",
                        "fg_color": "#D7E4BC",
                        "border": 1,
                    }
                )
                currency_fmt = wb.add_format({"num_format": "$#,##0.00"})
                date_fmt = wb.add_format({"num_format": "yyyy-mm-dd"})

                # --------------------------------------------------------------
                # Sales detail
                # --------------------------------------------------------------
                df.to_excel(writer, sheet_name="Sales Detail", index=False)
                ws = writer.sheets["Sales Detail"]

                for col, name in enumerate(df.columns):
                    ws.write(0, col, name, header_fmt)

                ws.set_column("A:A", 12)
                ws.set_column("B:B", 25)
                ws.set_column("C:C", 12)
                ws.set_column("D:D", 30)
                ws.set_column("E:Z", 15)

                for money in {
                    "Gross Value",
                    "Discount",
                    "Subtotal",
                    "Tax Charge",
                    "Total",
                } & set(df.columns):
                    idx = df.columns.get_loc(money)
                    ws.set_column(idx, idx, 15, currency_fmt)

                if "Date" in df.columns:
                    idx = df.columns.get_loc("Date")
                    ws.set_column(idx, idx, 12, date_fmt)

                # --------------------------------------------------------------
                # Summaries & trends
                # --------------------------------------------------------------
                self._create_sales_summary(wb, df)
                self._create_product_sales_summary(wb, df)
                if {"Customer ID", "Customer Name"} <= set(df.columns):
                    self._create_customer_sales_summary(wb, df)
                self._create_monthly_sales_trends(wb, df)

            logger.info("Sales report generated: %s", filepath)
            return filepath

        except Exception as exc:
            logger.exception("Error generating sales report: %s", exc)
            raise

    # --------------------------------------------------------------------- #
    #  FORECAST REPORT                                                      #
    # --------------------------------------------------------------------- #

    def generate_forecast_report(
        self,
        forecast_df: pd.DataFrame,
        inventory_df: pd.DataFrame | None = None,
    ) -> str:
        """
        Build an Excel file with forecast summaries and, when available,
        an inventory-vs-forecast comparison.

        Returns
        -------
        str
            Path to the created workbook.
        """
        if forecast_df.empty:
            raise ValueError("Cannot generate report: Forecast data is empty")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"forecast_report_{timestamp}.xlsx"
        filepath = os.path.join(self.temp_dir, filename)

        try:
            with pd.ExcelWriter(filepath, engine="xlsxwriter") as writer:
                wb = writer.book

                header_fmt = wb.add_format(
                    {
                        "bold": True,
                        "text_wrap": True,
                        "valign": "top",
                        "fg_color": "#D7E4BC",
                        "border": 1,
                    }
                )
                currency_fmt = wb.add_format({"num_format": "$#,##0.00"})
                date_fmt = wb.add_format({"num_format": "yyyy-mm-dd"})

                # --------------------------------------------------------------
                # Forecast summary
                # --------------------------------------------------------------
                forecast_df.to_excel(writer, sheet_name="Forecast Summary", index=False)
                ws = writer.sheets["Forecast Summary"]

                for col, name in enumerate(forecast_df.columns):
                    ws.write(0, col, name, header_fmt)

                ws.set_column("A:A", 12)
                ws.set_column("B:B", 15)
                ws.set_column("C:Z", 15)

                # --------------------------------------------------------------
                # Extra sheets
                # --------------------------------------------------------------
                self._create_forecast_by_period(wb, forecast_df)
                if inventory_df is not None and not inventory_df.empty:
                    self._create_inventory_vs_forecast(wb, inventory_df, forecast_df)

            logger.info("Forecast report generated: %s", filepath)
            return filepath

        except Exception as exc:
            logger.exception("Error generating forecast report: %s", exc)
            raise

    # ===================================================================== #
    #  INTERNAL SHEET-BUILDING HELPERS                                       #
    # ===================================================================== #

    def _add_inventory_dashboard(
        self,
        wb: xlsxwriter.Workbook,
        inventory_df: pd.DataFrame,
        stockout_df: pd.DataFrame | None = None,
    ) -> None:
        """Insert a visual dashboard into the workbook."""
        ws = wb.add_worksheet("Inventory Dashboard")

        title_fmt = wb.add_format(
            {"bold": True, "font_size": 16, "align": "center", "valign": "vcenter"}
        )
        ws.merge_range("A1:H1", "Inventory Dashboard", title_fmt)

        total_items = len(inventory_df)
        total_stock = (
            inventory_df["Current Stock"].sum()
            if "Current Stock" in inventory_df.columns
            else 0
        )
        low_stock = 0
        low_pct = 0.0
        if {"Current Stock", "Minimum Stock"} <= set(inventory_df.columns):
            low_stock = (inventory_df["Current Stock"] <= inventory_df["Minimum Stock"]).sum()
            low_pct = (low_stock / total_items * 100) if total_items else 0.0

        high = med = low = 0
        if (
            stockout_df is not None
            and not stockout_df.empty
            and "Stockout Risk" in stockout_df.columns
        ):
            high = (stockout_df["Stockout Risk"] == "High").sum()
            med = (stockout_df["Stockout Risk"] == "Medium").sum()
            low = (stockout_df["Stockout Risk"] == "Low").sum()

        lbl_fmt = wb.add_format({"font_size": 12, "align": "left"})
        val_fmt = wb.add_format({"font_size": 12, "align": "center", "bold": True})

        ws.write("A3", "Inventory Summary", lbl_fmt)
        for row, (text, val) in enumerate(
            [
                ("Total Items:", total_items),
                ("Total Stock:", total_stock),
                ("Low Stock Items:", low_stock),
                ("Low Stock %:", f"{low_pct:.1f}%"),
            ],
            start=4,
        ):
            ws.write(row, 0, text, lbl_fmt)
            ws.write(row, 1, val, val_fmt)

        ws.write("D3", "Stockout Risk Summary", lbl_fmt)
        for row, (text, val, fmt) in enumerate(
            [
                ("High Risk Items:", high, {"bg_color": "#FFC7CE", "font_color": "#9C0006"}),
                ("Medium Risk Items:", med, {"bg_color": "#FFEB9C", "font_color": "#9C5700"}),
                ("Low Risk Items:", low, {"bg_color": "#C6EFCE", "font_color": "#006100"}),
                (
                    "No Risk Items:",
                    total_items - (high + med + low),
                    {},
                ),
            ],
            start=4,
        ):
            ws.write(row, 3, text, lbl_fmt)
            ws.write(row, 4, val, wb.add_format({"font_size": 12, "align": "center", "bold": True, **fmt}))

        # Pie chart
        if high + med + low:
            chart = wb.add_chart({"type": "pie"})
            chart.add_series(
                {
                    "name": "Stockout Risk Distribution",
                    "categories": ["Inventory Dashboard", 3, 3, 6, 3],
                    "values": ["Inventory Dashboard", 3, 4, 6, 4],
                    "points": [
                        {"fill": {"color": "#FFC7CE"}},
                        {"fill": {"color": "#FFEB9C"}},
                        {"fill": {"color": "#C6EFCE"}},
                        {"fill": {"color": "#DDDDDD"}},
                    ],
                    "data_labels": {"value": True, "percentage": True},
                }
            )
            chart.set_title({"name": "Stockout Risk Distribution"})
            ws.insert_chart("A10", chart, {"x_scale": 1.5, "y_scale": 1.5})

    # ------------------------------------------------------------------ #

    def _create_sales_summary(self, wb: xlsxwriter.Workbook, df: pd.DataFrame) -> None:
        ws = wb.add_worksheet("Sales Summary")

        header_fmt = wb.add_format(
            {"bold": True, "text_wrap": True, "valign": "top", "fg_color": "#D7E4BC", "border": 1}
        )
        currency_fmt = wb.add_format({"num_format": "$#,##0.00"})
        title_fmt = wb.add_format({"bold": True, "font_size": 16, "align": "center", "valign": "vcenter"})

        ws.merge_range("A1:F1", "Sales Summary", title_fmt)

        total_inv = df["Invoice ID"].nunique() if "Invoice ID" in df.columns else 0
        total_units = df["Units Sold"].sum() if "Units Sold" in df.columns else 0
        total_sales = df["Total"].sum() if "Total" in df.columns else 0
        aov = total_sales / total_inv if total_inv else 0

        date_range = ""
        if "Date" in df.columns and not df["Date"].isna().all():
            date_range = f"From {df['Date'].min():%Y-%m-%d} to {df['Date'].max():%Y-%m-%d}"
        ws.merge_range("A2:F2", date_range, wb.add_format({"font_size": 12}))

        lbl_fmt = wb.add_format({"font_size": 12, "align": "left"})
        val_fmt = wb.add_format({"font_size": 12, "align": "center", "bold": True})

        for row, (label, value, fmt) in enumerate(
            [
                ("Total Invoices:", total_inv, val_fmt),
                ("Total Units Sold:", total_units, val_fmt),
                ("Total Sales:", total_sales, currency_fmt),
                ("Average Order Value:", aov, currency_fmt),
            ],
            start=4,
        ):
            ws.write(row, 0, label, lbl_fmt)
            ws.write(row, 1, value, fmt)

        # Monthly table + chart
        if "Date" in df.columns:
            df["Year"] = df["Date"].dt.year
            df["Month"] = df["Date"].dt.month
            monthly = (
                df.groupby(["Year", "Month"])
                .agg({"Units Sold": "sum", "Total": "sum", "Invoice ID": "nunique"})
                .reset_index()
            )
            month_names = [
                "January",
                "February",
                "March",
                "April",
                "May",
                "June",
                "July",
                "August",
                "September",
                "October",
                "November",
                "December",
            ]
            monthly["Period"] = (
                monthly["Month"].apply(lambda m: month_names[m - 1])
                + " "
                + monthly["Year"].astype(str)
            )

            ws.write("A10", "Monthly Sales Summary", title_fmt)

            headers = ["Period", "Units Sold", "Total Sales", "Number of Orders"]
            for col, h in enumerate(headers):
                ws.write(11, col, h, header_fmt)

            for i, row in enumerate(monthly.itertuples(), start=12):
                ws.write(i, 0, row.Period)
                ws.write(i, 1, row._3)  # Units Sold
                ws.write(i, 2, row.Total, currency_fmt)
                ws.write(i, 3, row.Invoice_ID)

            chart = wb.add_chart({"type": "column"})
            n_rows = len(monthly)
            chart.add_series(
                {
                    "name": "Monthly Sales",
                    "categories": ["Sales Summary", 12, 0, 11 + n_rows, 0],
                    "values": ["Sales Summary", 12, 2, 11 + n_rows, 2],
                    "data_labels": {"value": True},
                }
            )
            chart.set_title({"name": "Monthly Sales Trend"})
            chart.set_x_axis({"name": "Month"})
            chart.set_y_axis({"name": "Sales Value"})
            ws.insert_chart("E4", chart, {"x_scale": 1.5, "y_scale": 1.5})

    # ------------------------------------------------------------------ #

    def _create_product_sales_summary(self, wb: xlsxwriter.Workbook, df: pd.DataFrame) -> None:
        ws = wb.add_worksheet("Product Sales")

        header_fmt = wb.add_format(
            {"bold": True, "text_wrap": True, "valign": "top", "fg_color": "#D7E4BC", "border": 1}
        )
        currency_fmt = wb.add_format({"num_format": "$#,##0.00"})
        title_fmt = wb.add_format({"bold": True, "font_size": 16, "align": "center", "valign": "vcenter"})

        ws.merge_range("A1:F1", "Product Sales Analysis", title_fmt)

        if {"Product Code (SKU)", "Product Name"} <= set(df.columns):
            prod = (
                df.groupby(["Product Code (SKU)", "Product Name"])
                .agg({"Units Sold": "sum", "Total": "sum", "Invoice ID": "nunique"})
                .reset_index()
            )
            prod["Average Price"] = prod["Total"] / prod["Units Sold"]
            prod.sort_values("Total", ascending=False, inplace=True)

            headers = ["SKU", "Product Name", "Units Sold", "Total Sales", "Avg Price", "Order Count"]
            for col, h in enumerate(headers):
                ws.write(2, col, h, header_fmt)

            ws.set_column("A:A", 12)
            ws.set_column("B:B", 30)
            ws.set_column("C:F", 15)

            for i, row in enumerate(prod.itertuples(), start=3):
                ws.write(i, 0, row._1)
                ws.write(i, 1, row._2)
                ws.write(i, 2, row.Units_Sold)
                ws.write(i, 3, row.Total, currency_fmt)
                ws.write(i, 4, row.Average_Price, currency_fmt)
                ws.write(i, 5, row.Invoice_ID)

            chart = wb.add_chart({"type": "pie"})
            top_n = min(5, len(prod))
            chart.add_series(
                {
                    "name": "Top Products by Sales",
                    "categories": ["Product Sales", 3, 1, 2 + top_n, 1],
                    "values": ["Product Sales", 3, 3, 2 + top_n, 3],
                    "data_labels": {"percentage": True},
                }
            )
            chart.set_title({"name": "Top Products by Sales"})
            ws.insert_chart("H3", chart, {"x_scale": 1.5, "y_scale": 1.5})

    # ------------------------------------------------------------------ #

    def _create_customer_sales_summary(self, wb: xlsxwriter.Workbook, df: pd.DataFrame) -> None:
        ws = wb.add_worksheet("Customer Sales")

        header_fmt = wb.add_format(
            {"bold": True, "text_wrap": True, "valign": "top", "fg_color": "#D7E4BC", "border": 1}
        )
        currency_fmt = wb.add_format({"num_format": "$#,##0.00"})
        title_fmt = wb.add_format({"bold": True, "font_size": 16, "align": "center", "valign": "vcenter"})

        ws.merge_range("A1:F1", "Customer Sales Analysis", title_fmt)

        if {"Customer ID", "Customer Name"} <= set(df.columns):
            cust = (
                df.groupby(["Customer ID", "Customer Name"])
                .agg({"Units Sold": "sum", "Total": "sum", "Invoice ID": "nunique"})
                .reset_index()
            )
            cust["Average Order Value"] = cust["Total"] / cust["Invoice ID"]
            cust.sort_values("Total", ascending=False, inplace=True)

            headers = ["Customer ID", "Customer Name", "Total Sales", "Order Count", "Avg Order Value", "Units Purchased"]
            for col, h in enumerate(headers):
                ws.write(2, col, h, header_fmt)

            ws.set_column("A:A", 12)
            ws.set_column("B:B", 30)
            ws.set_column("C:F", 15)

            for i, row in enumerate(cust.itertuples(), start=3):
                ws.write(i, 0, row._1)
                ws.write(i, 1, row._2)
                ws.write(i, 2, row.Total, currency_fmt)
                ws.write(i, 3, row.Invoice_ID)
                ws.write(i, 4, row.Average_Order_Value, currency_fmt)
                ws.write(i, 5, row.Units_Sold)

    # ------------------------------------------------------------------ #

    def _create_monthly_sales_trends(self, wb: xlsxwriter.Workbook, df: pd.DataFrame) -> None:
        if "Date" not in df.columns:
            return

        ws = wb.add_worksheet("Monthly Trends")
        header_fmt = wb.add_format(
            {"bold": True, "text_wrap": True, "valign": "top", "fg_color": "#D7E4BC", "border": 1}
        )
        currency_fmt = wb.add_format({"num_format": "$#,##0.00"})
        title_fmt = wb.add_format({"bold": True, "font_size": 16, "align": "center", "valign": "vcenter"})

        ws.merge_range("A1:F1", "Monthly Sales Trends", title_fmt)

        df["Year"] = df["Date"].dt.year
        df["Month"] = df["Date"].dt.month

        monthly = (
            df.groupby(["Year", "Month"])
            .agg({"Total": "sum", "Units Sold": "sum", "Invoice ID": "nunique"})
            .reset_index()
        )
        month_names = [
            "January",
            "February",
            "March",
            "April",
            "May",
            "June",
            "July",
            "August",
            "September",
            "October",
            "November",
            "December",
        ]
        monthly["Month Name"] = monthly["Month"].apply(lambda m: month_names[m - 1])
        monthly["Period"] = monthly["Year"].astype(str) + "-" + monthly["Month"].astype(str).str.zfill(2)

        headers = ["Period", "Year", "Month", "Month Name", "Total Sales", "Units Sold", "Order Count"]
        for col, h in enumerate(headers):
            ws.write(2, col, h, header_fmt)

        ws.set_column("A:D", 15)
        ws.set_column("E:G", 15)

        for i, row in enumerate(monthly.itertuples(), start=3):
            ws.write(i, 0, row.Period)
            ws.write(i, 1, row.Year)
            ws.write(i, 2, row.Month)
            ws.write(i, 3, row._4)  # Month Name
            ws.write(i, 4, row.Total, currency_fmt)
            ws.write(i, 5, row.Units_Sold)
            ws.write(i, 6, row.Invoice_ID)

        chart = wb.add_chart({"type": "line"})
        chart.add_series(
            {
                "name": "Total Sales",
                "categories": ["Monthly Trends", 3, 3, 2 + len(monthly), 3],
                "values": ["Monthly Trends", 3, 4, 2 + len(monthly), 4],
                "marker": {"type": "diamond", "size": 7},
                "line": {"width": 2.5},
            }
        )
        chart.add_series(
            {
                "name": "Units Sold",
                "categories": ["Monthly Trends", 3, 3, 2 + len(monthly), 3],
                "values": ["Monthly Trends", 3, 5, 2 + len(monthly), 5],
                "marker": {"type": "circle", "size": 7},
                "line": {"width": 2.5},
                "y2_axis": True,
            }
        )
        chart.set_title({"name": "Monthly Sales and Units Trends"})
        chart.set_x_axis({"name": "Month"})
        chart.set_y_axis({"name": "Sales ($)"})
        chart.set_y2_axis({"name": "Units Sold"})
        chart.set_style(11)
        ws.insert_chart("I3", chart, {"x_scale": 2, "y_scale": 1.5})

    # ------------------------------------------------------------------ #

    def _create_forecast_by_period(self, wb: xlsxwriter.Workbook, df: pd.DataFrame) -> None:
        ws = wb.add_worksheet("Forecast by Period")
        header_fmt = wb.add_format(
            {"bold": True, "text_wrap": True, "valign": "top", "fg_color": "#D7E4BC", "border": 1}
        )
        title_fmt = wb.add_format({"bold": True, "font_size": 16, "align": "center", "valign": "vcenter"})
        ws.merge_range("A1:F1", "Forecast by Period", title_fmt)

        if {"SKU", "Forecast Period", "Forecasted Units"} <= set(df.columns):
            pivot = (
                df.pivot_table(
                    index="SKU",
                    columns="Forecast Period",
                    values="Forecasted Units",
                    aggfunc="sum",
                )
                .fillna(0)
                .reset_index()
            )
            periods = [p for p in pivot.columns if p != "SKU"]

            ws.write(2, 0, "SKU", header_fmt)
            for i, period in enumerate(periods, start=1):
                ws.write(2, i, period, header_fmt)

            ws.set_column(0, 0, 15)
            ws.set_column(1, len(periods), 15)

            for r, row in enumerate(pivot.itertuples(), start=3):
                ws.write(r, 0, row.SKU)
                for c, period in enumerate(periods, start=1):
                    ws.write(r, c, getattr(row, period))

            # Top-sku chart
            chart = wb.add_chart({"type": "column"})
            top_n = min(5, len(pivot))
            for i, period in enumerate(periods, start=1):
                chart.add_series(
                    {
                        "name": period,
                        "categories": ["Forecast by Period", 3, 0, 2 + top_n, 0],
                        "values": ["Forecast by Period", 3, i, 2 + top_n, i],
                    }
                )
            chart.set_title({"name": "Forecast by Period for Top SKUs"})
            chart.set_x_axis({"name": "SKU"})
            chart.set_y_axis({"name": "Forecasted Units"})
            chart.set_style(11)
            ws.insert_chart(
                f"A{3 + len(pivot) + 2}", chart, {"x_scale": 2, "y_scale": 1.5}
            )

    # ------------------------------------------------------------------ #

    def _create_inventory_vs_forecast(
        self,
        wb: xlsxwriter.Workbook,
        inv_df: pd.DataFrame,
        fc_df: pd.DataFrame,
    ) -> None:
        ws = wb.add_worksheet("Inventory vs Forecast")
        header_fmt = wb.add_format(
            {"bold": True, "text_wrap": True, "valign": "top", "fg_color": "#D7E4BC", "border": 1}
        )
        low_fmt = wb.add_format({"bg_color": "#FFC7CE", "font_color": "#9C0006"})
        ok_fmt = wb.add_format({"bg_color": "#C6EFCE", "font_color": "#006100"})
        title_fmt = wb.add_format({"bold": True, "font_size": 16, "align": "center", "valign": "vcenter"})

        ws.merge_range("A1:H1", "Inventory vs. Forecast Analysis", title_fmt)

        if {"SKU", "Current Stock"} <= set(inv_df.columns) and {"SKU", "Forecasted Units"} <= set(fc_df.columns):
            total_fc = fc_df.groupby("SKU")["Forecasted Units"].sum().reset_index()
            total_fc.rename(columns={"Forecasted Units": "Total Forecast"}, inplace=True)

            cmp = (
                inv_df[["SKU", "Product", "Current Stock", "Minimum Stock"]]
                .merge(total_fc, on="SKU", how="left")
                .fillna({"Total Forecast": 0})
            )

            cmp["Coverage Ratio"] = cmp["Current Stock"] / cmp["Total Forecast"].replace(0, np.nan)
            cmp["Coverage Ratio"].replace([np.inf, -np.inf], 999, inplace=True)
            cmp["Status"] = "OK"
            cmp.loc[cmp["Current Stock"] < cmp["Total Forecast"], "Status"] = "Insufficient"
            cmp.loc[cmp["Current Stock"] == 0, "Status"] = "Out of Stock"

            order = {"Out of Stock": 0, "Insufficient": 1, "OK": 2}
            cmp["__order"] = cmp["Status"].map(order)
            cmp.sort_values(["__order", "Coverage Ratio"], inplace=True)
            cmp.drop(columns="__order", inplace=True)

            headers = ["SKU", "Product", "Current Stock", "Minimum Stock", "Total Forecast", "Coverage Ratio", "Status"]
            for col, h in enumerate(headers):
                ws.write(2, col, h, header_fmt)

            ws.set_column("A:A", 12)
            ws.set_column("B:B", 30)
            ws.set_column("C:G", 15)

            for i, row in enumerate(cmp.itertuples(), start=3):
                ws.write(i, 0, row.SKU)
                ws.write(i, 1, row.Product)
                ws.write(i, 2, row.Current_Stock)
                ws.write(i, 3, row.Minimum_Stock)
                ws.write(i, 4, row.Total_Forecast)

                coverage = "∞" if row.Coverage_Ratio >= 999 else f"{row.Coverage_Ratio:.2f}"
                ws.write(i, 5, coverage)

                fmt = low_fmt if row.Status != "OK" else ok_fmt
                ws.write(i, 6, row.Status, fmt)

            # Chart for critical items
            critical = cmp[cmp["Status"] == "Insufficient"].head(10)
            if not critical.empty:
                chart = wb.add_chart({"type": "bar"})
                chart.add_series(
                    {
                        "name": "Current Stock",
                        "categories": ["Inventory vs Forecast", 3, 1, 2 + len(critical), 1],
                        "values": ["Inventory vs Forecast", 3, 2, 2 + len(critical), 2],
                        "fill": {"color": "#9BBB59"},
                    }
                )
                chart.add_series(
                    {
                        "name": "Total Forecast",
                        "categories": ["Inventory vs Forecast", 3, 1, 2 + len(critical), 1],
                        "values": ["Inventory vs Forecast", 3, 4, 2 + len(critical), 4],
                        "fill": {"color": "#C0504D"},
                    }
                )
                chart.set_title({"name": "Current Stock vs Forecast for Critical Items"})
                chart.set_x_axis({"name": "Product"})
                chart.set_y_axis({"name": "Units"})
                chart.set_style(11)
                ws.insert_chart(f"A{3 + len(cmp) + 2}", chart, {"x_scale": 2, "y_scale": 1.5})
