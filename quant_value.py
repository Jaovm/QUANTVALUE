import pandas as pd
import numpy as np

def calc_value_composite(df_fund):
    metrics = ['P/L', 'EV/EBITDA', 'P/VP', 'DY', 'ROE']
    df = df_fund.copy()
    for m in metrics:
        if m not in df.columns:
            df[m] = np.nan
    ranks = {}
    for m in metrics:
        if m in ['DY', 'ROE']:
            ranks[m] = df[m].rank(ascending=False)
        else:
            ranks[m] = df[m].rank(ascending=True)
    vc_score = sum(ranks[m] for m in metrics) / len(metrics)
    return 1 - (vc_score - vc_score.min()) / (vc_score.max() - vc_score.min())

def piotroski_f_score_placeholder(ticker):
    np.random.seed(hash(ticker) % 10000)
    return np.random.randint(4, 9)
