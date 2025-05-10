import numpy as np
from scipy.optimize import minimize

def optimize_portfolio(expected_returns, cov_matrix, current_weights, min_weights, max_weights, risk_free=0.02):
    n = len(expected_returns)
    x0 = np.array(current_weights)
    cons = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
    bnds = tuple((min_weights[i], max_weights[i]) for i in range(n))

    def neg_sharpe(x):
        port_return = np.dot(x, expected_returns)
        port_vol = np.sqrt(np.dot(x, np.dot(cov_matrix, x)))
        return -(port_return - risk_free) / port_vol if port_vol > 0 else 1e6

    result = minimize(neg_sharpe, x0, method='SLSQP', bounds=bnds, constraints=cons)
    return result.x if result.success else x0
