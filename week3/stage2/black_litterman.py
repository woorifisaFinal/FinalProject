# Ref: https://hwangheek.github.io/2021/black-litterman/
import numpy as np
import pandas as pd
from numpy.linalg import inv

from scipy import optimize
import json

# Date를 index로 선정
# excess_returns = pd.read_csv("stage1_prediction_21.csv", index_col=0)
excess_returns = pd.read_csv("stage1/output/stage1_prediction_21.csv", index_col=0)

N = excess_returns.columns.size

# w_mkt # 섹터별 비중, 임의의 합이 1이 되는 값들로 선정
w_mkt = np.random.randint(100,size=excess_returns.shape[1])
w_mkt = w_mkt/w_mkt.sum()

Sigma = excess_returns.cov()


expected_portfolio_excess_return = excess_returns.mean().multiply(w_mkt).sum()
portfolio_variance = w_mkt.dot(Sigma).dot(w_mkt)
lambd = expected_portfolio_excess_return / portfolio_variance
# lambd


# Implied Excess Equilibrium Return Vector (Prior인듯)
Pi = lambd * Sigma.dot(w_mkt)

###### View 
# view 개수
K = 3

Q = np.array([0.02, 0.03, 0.01])
assert Q.shape == (K,)


P = np.array([
    [0, 0, 0, 0,    0,    0, 0, 0, 1, 0,  0],
    [0, 0, 0, 0,    0,    0, 0, 0, 0, 1, -1],
    [1, 0, 0, 0, -.75, -.25, 0, 0, 0, 0,  0]
])
assert P.shape == (K, N)


tau = 1  # 0이 아닌 임의의 값
Omega = tau*P.dot(Sigma).dot(P.T) * np.eye(K)
ER = Pi + tau*Sigma.dot(P.T).dot(inv(P.dot(tau*Sigma).dot(P.T) + Omega).dot(Q - P.dot(Pi)))

w_hat = inv(Sigma).dot(ER)
w_hat = pd.Series(w_hat / w_hat.sum(), index=excess_returns.columns)
##### print(w_hat)
# brazil       0.104230
# india        0.140859
# taiwan       0.126773
# nasdaq       0.016098
# japan        0.075263
# uk           0.102225
# gold         0.074454
# bond3        0.060368
# bond10       0.096491
# kospi        0.117758
# eurostock    0.085481
# dtype: float64

w_hat - w_mkt

implied_Omega = np.zeros((K, K))

C = np.array([0.75, 0.25, 0.50])
assert C.shape == (K,)

k=0
ER_k_100 = Pi + tau*Sigma.dot(P[None, k].T).dot(inv(P[None, k].dot(tau * Sigma).dot(P[None, k].T)).dot(Q[None, k] - P[None, k].dot(Pi)))
w_k_100 = inv(Sigma).dot(ER_k_100)
w_k_100 = pd.Series(w_k_100 / w_k_100.sum(), index=e
                    xcess_returns.columns)

w_k_implied = w_mkt + (w_k_100 - w_mkt) * C[k]

def fun(omega_k):
    # 기존 ER을 구하는 수식에서 P를 P[None, k]로, Q를 Q[None, k]로 대체한 수식입니다.
    ER_k = Pi + tau * Sigma.dot(P[None, k].T).dot(inv(P[None, k].dot(tau * Sigma).dot(P[None, k].T) + omega_k).dot(Q[None, k] - P[None, k].dot(Pi)))
    
    w_k = inv(Sigma).dot(ER_k)
    w_k = pd.Series(w_k / w_k.sum(), index=excess_returns.columns)
    
    diff = w_k_implied - w_k
    return diff.T.dot(diff)

implied_Omega[k][k] = optimize.minimize_scalar(
    fun=fun,
    bounds=(1e-8, 1e+12),
    method='bounded',
).x


ER_with_CL = Pi + tau * Sigma.dot(P.T).dot(inv(P.dot(tau * Sigma).dot(P.T) + implied_Omega).dot(Q - P.dot(Pi)))
w_hat_with_CL = inv(Sigma).dot(ER_with_CL)
w_hat_with_CL = pd.Series(w_hat_with_CL / w_hat_with_CL.sum(), index=excess_returns.columns)
# w_hat_with_CL
# 음수도 나오네.? 0으로 만들고 다시 합 1로 만들까?
# w_hat_with_CL.sum()


w_hat_with_CL.to_json("portfolio_allocation.json")
# w_hat_with_CL.to_json(opj(cfg.base.output_dir, "portfolio_allocation.json"))

# with open("portfolio_allocation.json", "wb") as f:
#     json.dump()

# 해당 파일 읽을 때
# with open("portfolio_allocation.json", "rb") as f:
#     pa = json.load(f)