"""
Improved Delay Volume Forecasting Module
Uses moving average + linear trend for better stability
"""
import pandas as pd

from sklearn.linear_model import LinearRegression

from src.data_loader import load_master_events


def get_monthly_delay_forecast(forecast_months: int = 3):
    """
    Returns monthly historical delay data + forecast for next N months.
    """
    df = load_master_events()

    # Prepare date column
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date'])

    if len(df) == 0:
        return {"message": "No valid date data available"}

    # Monthly aggregation
    df['month'] = df['date'].dt.to_period('M').astype(str)
    monthly = df.groupby('month').size().reset_index(name='actual_delay_count')
    monthly = monthly.sort_values('month')

    if len(monthly) < 3:
        return {
            "message": "Not enough data for reliable forecasting (need at least 3 months)",
            "historical_data": monthly.to_dict(orient='records')
        }

    # Add time index for regression
    monthly['time_index'] = range(len(monthly))

    # Linear Regression for trend
    X = monthly[['time_index']]
    y = monthly['actual_delay_count']

    model = LinearRegression()
    model.fit(X, y)

    # Forecast next months
    last_index = len(monthly)
    future_index = pd.DataFrame({'time_index': range(last_index, last_index + forecast_months)})
    forecast_values = model.predict(future_index)

    forecast_data = []
    for i, val in enumerate(forecast_values):
        forecast_data.append({
            "month": f"Month+{i+1}",
            "predicted_delay_count": max(0, round(val, 1))
        })

    # Simple 3-month moving average as additional reference
    if len(monthly) >= 3:
        moving_avg = monthly['actual_delay_count'].tail(3).mean()
    else:
        moving_avg = monthly['actual_delay_count'].mean()

    return {
        "historical_data": monthly[['month', 'actual_delay_count']].to_dict(orient='records'),
        "forecast": forecast_data,
        "moving_average_reference": round(moving_avg, 1),
        "trend_slope": round(model.coef_[0], 2),
        "forecast_months": forecast_months
    }