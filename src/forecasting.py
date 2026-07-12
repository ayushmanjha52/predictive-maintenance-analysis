"""
Delay Volume Forecasting Module

FIXES vs. previous version:

1. CRITICAL: the old version derived month labels from df['date'], then
   dropped every row where 'date' failed to parse. Confirmed on the real
   dataset: ALL 92 Feb-26 rows have a NaN date in master_events.csv (a
   separate upstream data-unification issue), so the old script would
   have silently deleted an entire month from the forecast with zero
   indication anything was missing.
   Fix: use the existing `month` column, which is assigned directly
   during data unification and doesn't depend on date parsing at all.

2. Sorting month labels like "Nov-25" with .sort_values() sorts them
   ALPHABETICALLY, not chronologically -- confirmed:
     alphabetical: ['Apr-26','Dec-25','Feb-26','Jan-26','Mar-26','May-26','Nov-25']
     actual order: ['Nov-25','Dec-25','Jan-26','Feb-26','Mar-26','Apr-26','May-26']
   Fix: order explicitly via config.MONTH_ORDER (categorical), never a
   plain string sort.

3. Forecasted raw EVENT COUNT, mixing field-device and non-field-device
   delays together, inconsistent with every other module in this project
   (Pareto, FMEA fusion, dashboard) which all use field-device delay
   MINUTES as the unit of analysis. Fix: default to field-device minutes,
   with event count available as a secondary, clearly-labeled metric.

4. No goodness-of-fit reported for the trend line. Fix: reports R^2
   explicitly and downgrades to "use the rolling average instead" when
   R^2 is low or too few months exist -- matches the honest finding from
   earlier in this project (R^2=0.05 on 7 monthly points).
"""
import logging
import pandas as pd
from sklearn.linear_model import LinearRegression

from config import MONTH_ORDER
from data_loader import load_master_events, primary_device

logger = logging.getLogger(__name__)

R2_TRUST_THRESHOLD = 0.5  # below this, the linear trend is flagged unreliable
MIN_MONTHS_FOR_TREND = 6  # fewer months than this and the trend is flagged as noise-prone


def get_monthly_delay_forecast(forecast_months: int = 3, field_device_only: bool = True,
                                metric: str = "minutes"):
    """
    Returns monthly historical delay data + forecast for next N months.

    field_device_only: if True (default), only counts events resolved to
        a specific field device -- matches the scope of the rest of this
        project. Set False to forecast ALL delay types (furnace, motor,
        crane, etc.) if that's genuinely what you want.
    metric: "minutes" (total delay burden, the primary unit used
        elsewhere in this project) or "count" (number of events).
    """
    df = load_master_events()
    n_total = len(df)

    if field_device_only:
        df = df.copy()
        df["_primary_device"] = df["field_device"].apply(primary_device)
        df = df[df["_primary_device"].notna()]
        logger.info(f"Filtered to {len(df)}/{n_total} field-device-resolved events "
                    f"(field_device_only=True)")

    if metric == "minutes":
        df = df.copy()
        df["mins"] = pd.to_numeric(df["mins"], errors="coerce")
        n_bad_mins = df["mins"].isna().sum()
        if n_bad_mins:
            logger.warning(f"{n_bad_mins} rows have non-numeric 'mins' and are excluded from the sum.")
        df = df.dropna(subset=["mins"])
        monthly = df.groupby("month")["mins"].sum().reset_index(name="actual_value")
        value_label = "delay_minutes"
    elif metric == "count":
        monthly = df.groupby("month").size().reset_index(name="actual_value")
        value_label = "event_count"
    else:
        raise ValueError("metric must be 'minutes' or 'count'")

    # FIX: order chronologically via MONTH_ORDER, never alphabetically.
    monthly["month"] = pd.Categorical(monthly["month"], categories=MONTH_ORDER, ordered=True)
    monthly = monthly.sort_values("month").dropna(subset=["month"]).reset_index(drop=True)

    if len(monthly) == 0:
        return {"message": "No valid monthly data available after filtering."}

    if len(monthly) < 3:
        return {
            "message": "Not enough data for reliable forecasting (need at least 3 months).",
            "historical_data": monthly.to_dict(orient="records"),
        }

    monthly["time_index"] = range(len(monthly))
    X = monthly[["time_index"]]
    y = monthly["actual_value"]

    model = LinearRegression()
    model.fit(X, y)
    r2 = model.score(X, y)

    last_index = len(monthly)
    future_index = pd.DataFrame({"time_index": range(last_index, last_index + forecast_months)})
    forecast_values = model.predict(future_index)
    forecast_data = [
        {"month": f"Month+{i+1}", f"predicted_{value_label}": max(0, round(val, 1))}
        for i, val in enumerate(forecast_values)
    ]

    moving_avg = monthly["actual_value"].tail(3).mean()

    trend_reliable = (r2 >= R2_TRUST_THRESHOLD) and (len(monthly) >= MIN_MONTHS_FOR_TREND)
    recommended = round(moving_avg, 1) if not trend_reliable else round(forecast_data[0][f"predicted_{value_label}"], 1)

    return {
        "metric": value_label,
        "historical_data": monthly[["month", "actual_value"]].rename(
            columns={"actual_value": value_label}).to_dict(orient="records"),
        "forecast": forecast_data,
        "moving_average_reference": round(moving_avg, 1),
        "trend_slope": round(model.coef_[0], 2),
        "trend_r_squared": round(r2, 3),
        "trend_reliable": trend_reliable,
        "recommended_forecast": recommended,
        "note": (
            f"Trend R^2={r2:.2f} on {len(monthly)} months -- "
            + ("below the reliability threshold, use moving_average_reference instead of the trend forecast."
               if not trend_reliable else "trend meets the reliability bar for this sample size.")
        ),
        "forecast_months": forecast_months,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    import json
    result = get_monthly_delay_forecast()
    print(json.dumps(result, indent=2, default=str))