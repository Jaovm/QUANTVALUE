import pandas as pd
import numpy as np

def calc_value_composite(df_fund):
    """Calcula Value Composite Score (VC2/VC6 simplificado)"""
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
    # Normaliza para 0-1 (quanto maior, melhor)
    return 1 - (vc_score - vc_score.min()) / (vc_score.max() - vc_score.min())

def piotroski_f_score_placeholder(ticker):
    """Simulação de Piotroski F-Score. Substitua por cálculo real se desejar."""
    np.random.seed(hash(ticker) % 10000)
    return np.random.randint(4, 9)