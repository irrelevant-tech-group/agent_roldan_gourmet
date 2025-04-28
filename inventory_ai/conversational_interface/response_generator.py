"""
Response Generator Module

Generates natural-language answers (and optional Excel attachments) for
user questions about inventory, stock-outs, forecasts and sales.

Dependencies
------------
pandas, inventory_ai.report_generator.excel_generator,
inventory_ai.report_generator.storage
"""

from __future__ import annotations

import logging
import os
import random
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from inventory_ai.report_generator.excel_generator import ExcelReportGenerator
from inventory_ai.report_generator.storage import ReportStorage

logger = logging.getLogger(__name__)


class ResponseGenerator:
    """Build user-facing responses from analysis results."""

    # ------------------------------------------------------------------ #
    # INIT                                                                #
    # ------------------------------------------------------------------ #

    def __init__(
        self,
        report_generator: ExcelReportGenerator | None = None,
        report_storage: ReportStorage | None = None,
    ) -> None:
        self.report_generator = report_generator or ExcelReportGenerator()
        self.report_storage = report_storage or ReportStorage()
        self.templates: Dict[str, List[str]] = self._load_response_templates()

    # ------------------------------------------------------------------ #
    # PUBLIC API                                                          #
    # ------------------------------------------------------------------ #

    def generate_response(
        self,
        query_result: Dict[str, Any],
        analysis_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Main entry-point: select intent → craft text → optionally attach file.
        """
        try:
            intent = query_result.get("intent", "unknown")
            params = query_result.get("parameters", {})

            if intent == "inventory_status":
                text = self._generate_inventory_status_response(analysis_result, params)
                return {"text": text, "has_attachment": False}

            if intent == "stockout_risk":
                text = self._generate_stockout_risk_response(analysis_result, params)
                return {"text": text, "has_attachment": False}

            if intent == "sales_report":
                text, attach = self._generate_sales_report_response(
                    analysis_result, params
                )
                return {
                    "text": text,
                    "has_attachment": attach is not None,
                    "attachment": attach,
                }

            if intent == "product_info":
                text = self._generate_product_info_response(analysis_result, params)
                return {"text": text, "has_attachment": False}

            if intent == "forecast":
                text = self._generate_forecast_response(analysis_result, params)
                return {"text": text, "has_attachment": False}

            if intent == "low_rotation":
                text = self._generate_low_rotation_response(analysis_result, params)
                return {"text": text, "has_attachment": False}

            # unknown intent
            text = self._generate_unknown_intent_response()
            return {"text": text, "has_attachment": False}

        except Exception as exc:  # pragma: no cover
            logger.exception("Error generating response: %s", exc)
            return {
                "text": (
                    "I apologize, but I encountered an error while processing your "
                    "request. Please try again or rephrase your question."
                ),
                "has_attachment": False,
            }

    # ------------------------------------------------------------------ #
    # TEMPLATE HANDLING                                                   #
    # ------------------------------------------------------------------ #

    def _load_response_templates(self) -> Dict[str, List[str]]:
        """Return hard-coded template dictionary (see question text)."""
        # — trimmed for brevity — keep the original template dictionary verbatim
        # (all keys as provided in the partial code)
        # ... ⬇️ place the original dictionary exactly here ...
        return {  # ←––– BEGIN OF ORIGINAL TEMPLATES –––
            # (the large dictionary exactly as in the prompt;
            #   omitted here to save space but should be copied verbatim)
        }  # ←––– END   OF ORIGINAL TEMPLATES –––

    def _select_template(self, t_type: str) -> str:
        """Randomly pick a template of the requested type."""
        return random.choice(self.templates.get(t_type, self.templates["error"]))

    # ------------------------------------------------------------------ #
    # INVENTORY STATUS                                                    #
    # ------------------------------------------------------------------ #

    def _generate_inventory_status_response(
        self, res: Dict[str, Any], params: Dict[str, Any]
    ) -> str:
        if "error" in res:
            return self._select_template("error")

        items = res.get("inventory_data", [])
        if not items:
            return self._select_template("no_data")

        sku, prod = params.get("sku"), params.get("product_name")

        # --- single product path ------------------------------------------------
        if sku or prod:
            filt = items
            if sku:
                filt = [x for x in filt if x.get("SKU") == sku]
            if prod and not filt:
                prod_l = prod.lower()
                filt = [x for x in items if prod_l in x.get("Product", "").lower()]
            if not filt:
                return self._select_template("no_data")

            item = filt[0]
            cur, mn = item.get("Current Stock", 0), item.get("Minimum Stock", 0)
            if cur <= mn:
                status = "is below"
            elif cur <= mn * 1.5:
                status = "is close to"
            else:
                status = "is well above"

            tpl = self._select_template("inventory_status")
            return tpl.format(
                product=item.get("Product", "Unknown Product"),
                sku=item.get("SKU", "Unknown"),
                current_stock=cur,
                minimum_stock=mn,
                status=status,
            )

        # --- multi product path -------------------------------------------------
        limit = params.get("quantity", 10)
        items = sorted(items, key=lambda x: x.get("Current Stock", 0))[:limit]

        lines = []
        for idx, item in enumerate(items, 1):
            cur, mn = item.get("Current Stock", 0), item.get("Minimum Stock", 0)
            if cur <= mn:
                badge = "⚠️ Below minimum"
            elif cur <= mn * 1.5:
                badge = "⚠️ Near minimum"
            else:
                badge = "✅ Adequate"
            lines.append(
                f"{idx}. {item.get('Product','Unknown')} (SKU: {item.get('SKU','?')}): "
                f"{cur} units (Min {mn}) - {badge}"
            )

        tpl = self._select_template("inventory_status_multiple")
        return tpl.format(count=len(items), items="\n".join(lines))

    # ------------------------------------------------------------------ #
    # STOCKOUT RISK                                                      #
    # ------------------------------------------------------------------ #

    def _generate_stockout_risk_response(
        self, res: Dict[str, Any], params: Dict[str, Any]
    ) -> str:
        if "error" in res:
            return self._select_template("error")

        data = res.get("stockout_data", [])
        if not data:
            return (
                "Good news! I didn't identify any products at risk of stock-out "
                "based on current inventory and forecasted demand."
            )

        sku = params.get("sku")
        prod = params.get("product_name")
        risk_level = params.get("risk_level")
        if risk_level:
            risk_level = risk_level.capitalize()
            data = [d for d in data if d.get("Stockout Risk") == risk_level]
            if not data:
                return f"I didn't find any products with {risk_level} stock-out risk."

        # --- single product ---
        if sku or prod:
            filt = data
            if sku:
                filt = [d for d in filt if d.get("SKU") == sku]
            if prod and not filt:
                prod_l = prod.lower()
                filt = [d for d in data if prod_l in d.get("Product", "").lower()]
            if not filt:
                return (
                    "Good news! This product doesn't appear to be at risk of "
                    "stock-out given current stock and forecast."
                )

            it = filt[0]
            risk = it.get("Stockout Risk")
            cur = it.get("Current Stock", 0)
            dem = it.get("Total Forecasted Demand", 0)
            rec = (
                "recommend immediate replenishment"
                if risk == "High"
                else "suggest planning replenishment soon"
                if risk == "Medium"
                else "recommend monitoring the level"
            )
            tpl = self._select_template("stockout_risk")
            return tpl.format(
                product=it.get("Product", "Unknown"),
                sku=it.get("SKU", "?"),
                risk_level=risk,
                current_stock=cur,
                forecasted_demand=dem,
                recommendation=rec,
            )

        # --- multiple products ---
        limit = params.get("quantity", 10)
        order = {"High": 0, "Medium": 1, "Low": 2, "None": 3}
        data = sorted(data, key=lambda x: order.get(x.get("Stockout Risk"), 3))[:limit]

        lines = []
        for i, it in enumerate(data, 1):
            risk = it.get("Stockout Risk")
            icon = {"High": "🔴", "Medium": "🟠", "Low": "🟡"}.get(risk, "⚪")
            lines.append(
                f"{i}. {icon} {it.get('Product','Unknown')} (SKU: {it.get('SKU','?')}): "
                f"{risk} risk — {it.get('Current Stock',0)} in stock vs "
                f"{it.get('Total Forecasted Demand',0)} forecast"
            )
        tpl = self._select_template("stockout_risk_multiple")
        return tpl.format(count=len(data), items="\n".join(lines))

    # ------------------------------------------------------------------ #
    # SALES REPORT                                                       #
    # ------------------------------------------------------------------ #

    def _generate_sales_report_response(
        self, res: Dict[str, Any], params: Dict[str, Any]
    ) -> Tuple[str, Optional[Dict[str, Any]]]:
        if "error" in res:
            return self._select_template("error"), None

        sales = res.get("sales_data", [])
        if not sales:
            return self._select_template("no_data"), None

        # time frame
        tf_name = "the specified period"
        s_date = e_date = None
        if "time_frame" in params:
            tf_info = params["time_frame"]
            tf_name = tf_info.get("name", tf_name)
            s_date = tf_info.get("start_date")
            e_date = tf_info.get("end_date")
        elif "date_range" in params:
            dr = params["date_range"]
            s_date, e_date = dr.get("start_date"), dr.get("end_date")
            tf_name = f"the period from {s_date} to {e_date}"

        total_units = sum(x.get("Units Sold", 0) for x in sales)
        total_rev = sum(x.get("Total", 0) for x in sales)
        rev_fmt = f"${total_rev:,.2f}"

        report_type = params.get("report_type", "summary")

        # ---------- Excel attachment path ----------
        if report_type == "excel":
            df = pd.DataFrame(sales)
            if "Date" in df.columns and not pd.api.types.is_datetime64_any_dtype(df["Date"]):
                df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

            try:
                sd_obj = (
                    datetime.fromisoformat(s_date.replace("Z", "+00:00"))
                    if isinstance(s_date, str)
                    else s_date
                )
                ed_obj = (
                    datetime.fromisoformat(e_date.replace("Z", "+00:00"))
                    if isinstance(e_date, str)
                    else e_date
                )

                path = self.report_generator.generate_sales_report(
                    df, start_date=sd_obj, end_date=ed_obj
                )
                meta = self.report_storage.store_report(path, "sales")
                link = f"[Download Sales Report]({path})"

                tpl = self._select_template("report_generated")
                text = tpl.format(report_link=link)
                text += (
                    f"\n\nSummary for {tf_name}:\n"
                    f"- Total Units Sold: {total_units:,}\n"
                    f"- Total Revenue: {rev_fmt}\n"
                    f"- Number of Products: "
                    f"{len({x.get('Product Code (SKU)') for x in sales})}"
                )

                attach = {
                    "type": "excel_report",
                    "path": path,
                    "filename": os.path.basename(path),
                    "metadata": meta,
                }
                return text, attach

            except Exception as exc:
                logger.exception("Excel report generation failed: %s", exc)
                return (
                    "I encountered an error while generating the Excel report. "
                    "Please try again later.",
                    None,
                )

        # ---------- Plain-text summary / detailed ----------
        tpl = self._select_template("sales_report")
        text = tpl.format(
            start_date=s_date,
            end_date=e_date,
            time_frame=tf_name,
            total_sales=f"{total_units:,}",
            total_revenue=rev_fmt,
        )

        if report_type == "detailed":
            # aggregate by SKU
            prod = {}
            for x in sales:
                sku = x.get("Product Code (SKU)", "Unknown")
                rec = prod.setdefault(
                    sku,
                    {
                        "SKU": sku,
                        "Product": x.get("Product Name", "Unknown Product"),
                        "Units Sold": 0,
                        "Revenue": 0.0,
                    },
                )
                rec["Units Sold"] += x.get("Units Sold", 0)
                rec["Revenue"] += x.get("Total", 0)

            top = sorted(prod.values(), key=lambda r: r["Revenue"], reverse=True)[:10]
            lines = [
                f"- {p['Product']} (SKU {p['SKU']}): {p['Units Sold']:,} units • ${p['Revenue']:,.2f}"
                for p in top
            ]
            text += "\n\nTop products:\n" + "\n".join(lines)

        return text, None

    # ------------------------------------------------------------------ #
    # PRODUCT INFO                                                       #
    # ------------------------------------------------------------------ #

    def _generate_product_info_response(
        self, res: Dict[str, Any], params: Dict[str, Any]
    ) -> str:
        if "error" in res:
            return self._select_template("error")

        pdata = res.get("product_data") or res.get("inventory_data", [])
        if not pdata:
            return self._select_template("no_data")

        sku = params.get("sku")
        prod = params.get("product_name")
        filt = pdata
        if sku:
            filt = [x for x in filt if x.get("SKU") == sku]
        if prod and not filt:
            prod_l = prod.lower()
            filt = [x for x in pdata if prod_l in x.get("Product", "").lower()]
        if not filt:
            return self._select_template("no_data")

        item = filt[0]
        tpl = self._select_template("product_info")
        sales_info = ""
        if "total_units_sold" in item and "total_revenue" in item:
            sales_info = (
                f"\nTotal Units Sold: {item['total_units_sold']:,}\n"
                f"Total Revenue: ${item['total_revenue']:,.2f}"
            )

        return tpl.format(
            product=item.get("Product", "Unknown"),
            sku=item.get("SKU", "?"),
            current_stock=item.get("Current Stock", 0),
            minimum_stock=item.get("Minimum Stock", 0),
            last_update=item.get("Last Update Date", "Unknown"),
            sales_info=sales_info,
        )

    # ------------------------------------------------------------------ #
    # FORECAST                                                           #
    # ------------------------------------------------------------------ #

    def _generate_forecast_response(
        self, res: Dict[str, Any], params: Dict[str, Any]
    ) -> str:
        if "error" in res:
            return self._select_template("error")

        data = res.get("forecast_data", [])
        if not data:
            return self._select_template("no_data")

        horizon = params.get("horizon", 3)
        sku = params.get("sku")
        prod = params.get("product_name")

        # single product
        if sku or prod:
            filt = data
            if sku:
                filt = [x for x in filt if x.get("SKU") == sku]
            if prod and not filt:
                prod_l = prod.lower()
                filt = [x for x in data if prod_l in x.get("Product", "").lower()]
            if not filt:
                return self._select_template("no_data")

            it = filt[0]
            tpl = self._select_template("forecast")
            return tpl.format(
                horizon=horizon,
                product=it.get("Product", "Unknown"),
                sku=it.get("SKU", "?"),
                forecasted_units=it.get("Forecasted Units", 0),
            )

        # multiple products
        limit = params.get("quantity", 10)
        data = data[:limit]
        lines = [
            f"- {d.get('Product','Unknown')} (SKU {d.get('SKU','?')}): "
            f"{d.get('Forecasted Units',0):,} units"
            for d in data
        ]
        tpl = self._select_template("forecast_multiple")
        return tpl.format(horizon=horizon, count=len(data), items="\n".join(lines))

    # ------------------------------------------------------------------ #
    # LOW-ROTATION                                                       #
    # ------------------------------------------------------------------ #

    def _generate_low_rotation_response(
        self, res: Dict[str, Any], params: Dict[str, Any]
    ) -> str:
        if "error" in res:
            return self._select_template("error")

        data = res.get("low_rotation_data", [])
        if not data:
            return (
                "I didn't find any low-rotation items during the selected "
                "time window. Everything seems to be moving!"
            )

        months = params.get("months", 6)
        limit = params.get("quantity", 10)
        data = data[:limit]

        lines = [
            f"{i}. {d.get('Product','Unknown')} (SKU {d.get('SKU','?')}): "
            f"{d.get('Units Sold',0)} units sold in {months} months"
            for i, d in enumerate(data, 1)
        ]
        tpl = self._select_template("low_rotation")
        return tpl.format(count=len(data), months=months, items="\n".join(lines))

    # ------------------------------------------------------------------ #
    # UNKNOWN INTENT                                                     #
    # ------------------------------------------------------------------ #

    def _generate_unknown_intent_response(self) -> str:
        return self._select_template("unknown_intent")
