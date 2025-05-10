import pandas as pd
import numpy as np
import statsmodels.api as sm
from arch import arch_model

def estimate_arima_forecast(prices, steps=252):
    returns = prices.pct_change().dropna()
    forecasts = {}
    for col in returns:
        try:
            model = sm.tsa.ARIMA(returns[col], order=(1, 0, 0)).fit()
            pred = model.forecast(steps=steps)
            forecasts[col] = pred.mean()
        except Exception:
            forecasts[col] = returns[col].mean()
    return pd.Series(forecasts)

def estimate_garch_volatility(prices):
    returns = prices.pct_change().dropna()
    annualized_vols = {}
    for col in returns:
        try:
            am = arch_model(returns[col]*100, vol='Garch', p=1, o=0, q=1, dist='normal')
            res = am.fit(disp='off')
            forecast = res.forecast(horizon=1)
            cond_vol = forecast.variance.values[-1, 0] ** 0.5 / 100
            annualized_vols[col] = cond_vol * np.sqrt(252)
        except Exception:
            annualized_vols[col] = returns[col].std() * np.sqrt(252)
    return pd.Series(annualized_vols)
