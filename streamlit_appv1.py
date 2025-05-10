import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

from financial_analyzerv2 import (
    download_prices,
    get_fundamentals_yahoo,
    obter_metricas_avancadas,
    otimizar_portfolio_institucional,
)

st.set_page_config(layout="wide")
st.title("QUANTOVITZ 2.0 — Otimização Institucional de Carteiras")

st.sidebar.header("Entradas do Usuário")

st.sidebar.subheader("1. Carteira Atual")
ativos_input_str = st.sidebar.text_input("Ativos da carteira (ex: PETR4.SA,VALE3.SA,ITUB4.SA)", "PETR4.SA,VALE3.SA,ITUB4.SA")
pesos_input_str = st.sidebar.text_input("Pesos percentuais (ex: 40,30,30)", "40,30,30")
valor_total_carteira_atual = st.sidebar.number_input("Valor total da carteira atual (R$)", min_value=0.0, value=100000.0, step=1000.0)

st.sidebar.subheader("2. Novo Aporte (Opcional)")
novo_capital_input = st.sidebar.number_input("Novo capital a ser aportado (R$)", min_value=0.0, value=10000.0, step=100.0)

st.sidebar.subheader("3. Ativos Candidatos (Opcional)")
candidatos_input_str = st.sidebar.text_input("Ativos candidatos à entrada (ex: MGLU3.SA,VIIA3.SA)", "MGLU3.SA,VIIA3.SA")

st.sidebar.subheader("4. Modelo de Otimização")
modelo_selecionado = st.sidebar.selectbox(
    "Escolha o modelo",
    ("Somente Quant-Value", "Quant-Value + Fronteira Eficiente", "Quant-Value + Fronteira Eficiente + Econometria")
)

st.sidebar.subheader("5. Pesos das Métricas Fundamentalistas")
pesos_metricas = {}
pesos_metricas['ROE'] = st.sidebar.slider("Peso ROE (%)", 0, 100, 25) / 100.0
pesos_metricas['EV/EBITDA'] = st.sidebar.slider("Peso EV/EBITDA (%)", 0, 100, 25) / 100.0
pesos_metricas['P/L'] = st.sidebar.slider("Peso P/L (%)", 0, 100, 25) / 100.0
pesos_metricas['DY'] = st.sidebar.slider("Peso DY (%)", 0, 100, 25) / 100.0

soma_pesos_metricas = sum(pesos_metricas.values())
if soma_pesos_metricas == 0:
    st.sidebar.warning("Defina pesos para as métricas fundamentalistas para usar o Quant-Value.")
elif abs(soma_pesos_metricas - 1.0) > 1e-6:
    st.sidebar.warning(f"A soma dos pesos das métricas ({soma_pesos_metricas*100:.0f}%) não é 100%. Eles serão normalizados.")

st.sidebar.subheader("6. Limites de Alocação por Ativo (Opcional)")
min_aloc_ativo = st.sidebar.slider("Alocação Mínima por Ativo (%)", 0, 100, 0) / 100.0
max_aloc_ativo = st.sidebar.slider("Alocação Máxima por Ativo (%)", 0, 100, 50) / 100.0

run_analysis = st.sidebar.button("Executar Análise")

def plot_quant_value_scores(df_scores):
    if df_scores.empty:
        st.write("Não há dados de Score Quant-Value para exibir.")
        return
    fig = px.bar(df_scores, x='Ticker', y='Quant Score', title='Score Quant-Value por Ativo',
                 labels={'Quant Score': 'Score (0-1)', 'Ticker': 'Ativo'},
                 color='Quant Score', color_continuous_scale=px.colors.sequential.Viridis)
    fig.update_layout(xaxis_title="Ativo", yaxis_title="Score Quant-Value")
    st.plotly_chart(fig, use_container_width=True)

def plot_portfolio_pie_chart(weights_dict, title):
    if not weights_dict or sum(weights_dict.values()) == 0:
        st.write(f"Não há dados para o gráfico de pizza: {title}")
        return
    df_pie = pd.DataFrame(list(weights_dict.items()), columns=['Ativo', 'Peso'])
    df_pie = df_pie[df_pie['Peso'] > 1e-4]
    fig = px.pie(df_pie, values='Peso', names='Ativo', title=title, hole=0.3)
    fig.update_traces(textposition='inside', textinfo='percent+label')
    st.plotly_chart(fig, use_container_width=True)

def display_comparative_table(carteiras_data):
    if not carteiras_data:
        st.write("Não há dados de carteiras para comparar.")
        return
    df_comparativo = pd.DataFrame(carteiras_data)
    df_comparativo_display = df_comparativo[['Nome', 'Retorno Esperado (%)', 'Volatilidade (%)', 'Sharpe Ratio']].copy()
    df_comparativo_display = df_comparativo_display.set_index('Nome')
    st.subheader("Tabela Comparativa de Carteiras")
    st.dataframe(df_comparativo_display.style.format("{:.2f}"))
    st.subheader("Composição Detalhada das Carteiras (%)")
    todos_ativos_pesos = set()
    for c in carteiras_data:
        if 'Pesos' in c and isinstance(c['Pesos'], dict):
            todos_ativos_pesos.update(c['Pesos'].keys())
    pesos_data_list = []
    for c in carteiras_data:
        row = {'Nome': c['Nome']}
        if 'Pesos' in c and isinstance(c['Pesos'], dict):
            for ativo in todos_ativos_pesos:
                row[ativo] = c['Pesos'].get(ativo, 0) * 100
        pesos_data_list.append(row)
    df_pesos_detalhados = pd.DataFrame(pesos_data_list).set_index('Nome')
    st.dataframe(df_pesos_detalhados.style.format("{:.2f}"))

def plot_efficient_frontier(mu, cov, min_w, max_w, n_points=40, mark_portfolios=None):
    mus = np.linspace(mu.min(), mu.max(), n_points)
    results = []
    n = len(mu)
    for target_return in mus:
        cons = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
                {'type': 'eq', 'fun': lambda x: np.dot(x, mu) - target_return}]
        bnds = tuple((min_w, max_w) for _ in range(n))
        x0 = np.ones(n) / n
        try:
            res = minimize(lambda x: np.sqrt(np.dot(x, np.dot(cov, x))), x0, bounds=bnds, constraints=cons)
        except Exception:
            continue
        if res is not None and res.success:
            results.append((res.fun, target_return, res.x))
    if not results:
        st.warning("Não foi possível gerar a Fronteira Eficiente. Tente adicionar mais ativos ou ajustar restrições.")
        return
    vols, rets, weights = zip(*results)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=np.array(vols)*100, y=np.array(rets)*100,
                             mode='lines+markers',
                             name='Fronteira Eficiente'))
    if mark_portfolios:
        for nome, (vol, ret, cor) in mark_portfolios.items():
            fig.add_trace(go.Scatter(x=[vol*100], y=[ret*100],
                                     mode='markers', name=nome,
                                     marker=dict(color=cor, size=13, symbol='star')))
    fig.update_layout(title='Fronteira Eficiente (Markowitz)',
        xaxis_title='Volatilidade Anualizada (%)', yaxis_title='Retorno Esperado Anualizado (%)')
    st.plotly_chart(fig, use_container_width=True)

if run_analysis:
    st.header("Resultados da Análise")

    ativos_carteira_lista = [s.strip().upper() for s in ativos_input_str.split(',') if s.strip()]
    try:
        pesos_carteira_lista_pct = [float(p.strip()) for p in pesos_input_str.split(',') if p.strip()]
        if len(ativos_carteira_lista) != len(pesos_carteira_lista_pct):
            st.error("O número de ativos e pesos na carteira atual deve ser o mesmo.")
            st.stop()
        if abs(sum(pesos_carteira_lista_pct) - 100.0) > 1e-2:
            st.warning(f"A soma dos pesos da carteira atual ({sum(pesos_carteira_lista_pct):.2f}%) não é 100%.")
        pesos_carteira_decimal = [p/100.0 for p in pesos_carteira_lista_pct]
        carteira_atual_composicao_valores = {ativo: peso_dec * valor_total_carteira_atual for ativo, peso_dec in zip(ativos_carteira_lista, pesos_carteira_decimal)}
    except ValueError:
        st.error("Os pesos da carteira atual devem ser números.")
        st.stop()

    ativos_candidatos_lista = [s.strip().upper() for s in candidatos_input_str.split(',') if s.strip()]
    todos_ativos_analise = list(dict.fromkeys(ativos_carteira_lista + ativos_candidatos_lista))

    if not todos_ativos_analise:
        st.error("Nenhum ativo fornecido para análise.")
        st.stop()

    with st.spinner("Baixando dados de preços e fundamentos..."):
        prices = download_prices(todos_ativos_analise, start="2015-01-01")
        df_fund = get_fundamentals_yahoo(todos_ativos_analise)
        if df_fund.empty:
            st.warning("Não foi possível obter dados fundamentalistas para os ativos.")
        else:
            st.write("**Dados Fundamentalistas:**")
            st.dataframe(df_fund)

    with st.spinner("Calculando métricas avançadas (Value, F-Score, ARIMA, GARCH)..."):
        vc_scores, f_score, mu_arima, vol_garch = obter_metricas_avancadas(todos_ativos_analise, prices, df_fund)
        df_scores = pd.DataFrame({'Ticker': todos_ativos_analise, 'Quant Score': vc_scores.values})
        plot_quant_value_scores(df_scores)

    st.write("**Retornos esperados (ARIMA ajustado):**")
    st.dataframe(mu_arima)
    st.write("**Volatilidade (GARCH):**")
    st.dataframe(vol_garch)

    mu_final = mu_arima * (1 + 0.3 * vc_scores) * (1 + 0.1 * f_score/9)
    mu_final = mu_final.fillna(0)
    vol_garch = vol_garch.fillna(vol_garch.mean())
    cov_final = np.diag(vol_garch.values**2)

    weights_current = []
    for t in todos_ativos_analise:
        if t in ativos_carteira_lista:
            idx = ativos_carteira_lista.index(t)
            weights_current.append(pesos_carteira_decimal[idx])
        else:
            weights_current.append(0.0)
    min_w = [max(min_aloc_ativo, 0.0) for _ in todos_ativos_analise]
    max_w = [max(max_aloc_ativo, min_aloc_ativo+0.01) for _ in todos_ativos_analise]
    if sum(min_w) > 1.0:
        st.warning("A soma das alocações mínimas excede 100%. Ajustando para 0.")
        min_w = [0.0 for _ in todos_ativos_analise]
    if sum(max_w) < 1.0:
        st.warning("A soma das alocações máximas é menor que 100%. Ajustando para 1/n.")
        max_w = [1.0/len(max_w) for _ in max_w]

    with st.spinner("Otimizando portfólio institucional..."):
        pesos_otimizados = otimizar_portfolio_institucional(
            todos_ativos_analise, weights_current, min_w, max_w, mu_final, cov_final, risk_free=0.02
        )
    st.subheader("Carteira Otimizada (Institucional)")
    plot_portfolio_pie_chart(pesos_otimizados, "Carteira Otimizada")
    st.write("**Pesos Otimizados:**")
    st.dataframe(pd.DataFrame({'Ativo': list(pesos_otimizados.keys()), 'Peso': list(pesos_otimizados.values())}))

    def carteira_metrics(pesos, mu, cov, taxa_rf=0.02):
        w = np.array([pesos.get(t,0) for t in mu.index])
        ret = np.dot(w, mu)
        vol = np.sqrt(np.dot(w, np.dot(cov, w)))
        sharpe = (ret-taxa_rf)/vol if vol > 0 else 0
        return ret, vol, sharpe

    ret_atual, vol_atual, sharpe_atual = carteira_metrics(dict(zip(todos_ativos_analise, weights_current)), mu_final, cov_final)
    ret_otim, vol_otim, sharpe_otim = carteira_metrics(pesos_otimizados, mu_final, cov_final)

    carteiras_para_comparacao = [
        {
            'Nome': 'Carteira Atual',
            'Retorno Esperado (%)': ret_atual*100,
            'Volatilidade (%)': vol_atual*100,
            'Sharpe Ratio': sharpe_atual,
            'Pesos': dict(zip(todos_ativos_analise, weights_current))
        },
        {
            'Nome': 'Carteira Otimizada',
            'Retorno Esperado (%)': ret_otim*100,
            'Volatilidade (%)': vol_otim*100,
            'Sharpe Ratio': sharpe_otim,
            'Pesos': pesos_otimizados
        }
    ]
    display_comparative_table(carteiras_para_comparacao)

    with st.spinner("Gerando gráfico da fronteira eficiente..."):
        mark_portfolios = {
            "Atual": (vol_atual, ret_atual, "blue"),
            "Otimizada": (vol_otim, ret_otim, "red"),
        }
        plot_efficient_frontier(mu_final, cov_final, min_aloc_ativo, max_aloc_ativo, n_points=20, mark_portfolios=mark_portfolios)

    if novo_capital_input > 0:
        st.subheader("Sugestão de Alocação para Novo Aporte")
        carteira_final_valores = carteira_atual_composicao_valores.copy()
        for ativo, peso in pesos_otimizados.items():
            atual = carteira_final_valores.get(ativo, 0)
            carteira_final_valores[ativo] = atual + novo_capital_input * peso
        soma_final = sum(carteira_final_valores.values())
        pesos_finais = {k: v/soma_final for k, v in carteira_final_valores.items()}
        plot_portfolio_pie_chart(pesos_finais, f"Carteira Após Novo Aporte (R$ {novo_capital_input:,.2f})")
else:
    st.info("Ajuste os parâmetros na barra lateral e clique em 'Executar Análise' para ver os resultados.")

st.sidebar.markdown("---")
st.sidebar.subheader("Glossário e Explicações")
with st.sidebar.expander("Quant-Value"):
    st.write("O modelo Quant-Value combina métricas fundamentalistas para atribuir um score a cada ativo, ajudando a identificar os mais atrativos com base em valor.")
with st.sidebar.expander("Fronteira Eficiente"):
    st.write("A Fronteira Eficiente de Markowitz mostra o conjunto de portfólios ótimos que oferecem o maior retorno esperado para um dado nível de risco, ou o menor risco para um dado retorno.")
with st.sidebar.expander("Índice de Sharpe"):
    st.write("O Índice de Sharpe mede o retorno de um investimento ajustado pelo risco. Um Sharpe mais alto indica melhor desempenho por unidade de risco.")
with st.sidebar.expander("Econometria"):
    st.write("Modelos econométricos (como ARIMA e GARCH) ajudam a prever retornos e volatilidades futuras, tornando a alocação mais robusta e previsível.")
