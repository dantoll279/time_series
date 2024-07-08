# What forecasting models shall we include? What libraries will we need to use these models?

# Example
import pandas as pd
from prophet import Prophet


def prophet_forecast(df, horizon, **kwargs):
    """
    :param df: Pandas.DataFrame - historical time series data.
    :param horizon: int - Number of time steps to forecast.
    :param kwargs: Facebook Prophet keyword arguments.
    :return: Pandas.DataFrame - Forecast.
    """

    model = Prophet(**kwargs)
    model.fit(df)
    future = model.make_future_dataframe(periods=horizon)

    forecast = model.predict(future)
    return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

