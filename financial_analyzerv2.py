import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime

from quant_value import calc_value_composite, piotroski_f_score_placeholder
from econometric_models import estimate_arima_forecast, estimate_garch_volatility
from portfolio_optimizer import optimize_portfolio

def download_prices(tickers, start="2015-01-01", end=None):
    if end is None:
        end = datetime.today().strftime('%Y-%m-%d')
    data = yf.download(tickers, start=start, end=end, progress=False)
    # Normaliza para DataFrame de preço ajustado (colunas = tickers)
    if isinstance(data.columns, pd.MultiIndex):
        if "Adj Close" in data.columns.levels[0]:
            data = data["Adj Close"]
        else:
            data = data["Close"]
    elif "Adj Close" in data.columns:
        data = data["Adj Close"]
    elif "Close" in data.columns:
        data = data["Close"]
    if isinstance(data, pd.Series):
        data = data.to_frame()
    data = data.dropna(how='all')
    data = data.loc[:, data.columns.intersection(tickers)]  # Garante ordem dos ativos
    return data

def get_fundamentals_yahoo(tickers):
    data = []
    for t in tickers:
        try:
            info = yf.Ticker(t).info
            data.append({
                'ticker': t,
                'ROE': info.get('returnOnEquity'),
                'EV/EBITDA': info.get('enterpriseToEbitda'),
                'P/L': info.get('trailingPE'),
                'P/VP': info.get('priceToBook'),
                'DY': info.get('dividendYield'),
                'NetMargin': info.get('netMargins'),
                'EPS_growth': info.get('earningsQuarterlyGrowth'),
            })
        except Exception:
            data.append({'ticker': t})
    return pd.DataFrame(data).set_index('ticker')

def obter_metricas_avancadas(tickers, prices, df_fund):
    vc_scores = calc_value_composite(df_fund)
    f_scores = pd.Series({t: piotroski_f_score_placeholder(t) for t in tickers}, index=tickers)
    mu_arima = estimate_arima_forecast(prices)
    vol_garch = estimate_garch_volatility(prices)
    return vc_scores, f_scores, mu_arima, vol_garch

def otimizar_portfolio_institucional(
    tickers,
    weights_current,
    min_w,
    max_w,
    mu_final,
    cov_final,
    risk_free=0.02
):
    opt_weights = optimize_portfolio(mu_final.values, cov_final, weights_current, min_w, max_w, risk_free)
    return dict(zip(tickers, opt_weights))
