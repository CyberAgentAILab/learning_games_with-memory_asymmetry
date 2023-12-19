"""
discretized multi-memory gradient ascent
(m,nx,ny)=(2,1,0): m=action number, nx=X's memory length, ny=Y's memory length
output textfile here as dMMGA_210_test.txt
"""

import numpy as np

det = 1  # 1: random initial state, 2: memory-less initial state
if det == 1:
    x_vc, y = np.random.random(4), np.random.random(1)
elif det == 2:
    x_vc, y = 0.6 * np.ones(5), 0.6 * np.ones(1)
xc_vc, yc = np.copy(x_vc), np.copy(y)
uo_vc, vo_vc = [1, -1, -1, 1], [-1, 1, 1, -1]
u_vc, v_vc = np.array(uo_vc), np.array(vo_vc)
Tmax, NdT = 10**3, 10**3
dp, maxdis = 10**-6, 10**-9


# eigenvector
def EIGEN_VECTOR(M_mt):
    distance = 1
    p_vc = np.ones(4) / 4
    while distance > maxdis:
        pn_vc = np.dot(M_mt, p_vc)
        pn_vc = pn_vc / np.sum(pn_vc)
        distance = np.sum((pn_vc - p_vc) ** 2) ** 0.5
        p_vc = pn_vc
    return p_vc


# matrix generation
def M_MATRIX(x_vc, y):
    M_mt = np.array([x_vc * y, x_vc * (1 - y), (1 - x_vc) * y, (1 - x_vc) * (1 - y)])
    return M_mt


# output textfile
txt = open("dMMGA_210_test.txt", "w")
txt.write(
    "#time:[Tmax,NdT] (Discretized Multi-memory Gradient Ascent) = "
    + str([Tmax, NdT])
    + "\n"
)
txt.write(
    "#detailed of gradient: [measure=log10(dp),accuracy=log10(maxdis)] = "
    + str([int(np.log10(dp)), int(np.log10(maxdis))])
    + "\n"
)
txt.write("#payoff:u = " + str(uo_vc) + ", v = " + str(vo_vc) + "\n")
txt.write("#strategies: one-memory vs zero-memory" + "\n")
txt.write("#t_vc(1), x_vc(4), y(1), p_vc(4), ust(1), vst(1)" + "\n")

xst_vc, yst_vc = [], []
ueqp_vc, veqp_vc = [], []
for t in range(0, int(Tmax * NdT) + 1):
    M_mt = M_MATRIX(x_vc, y)
    po_vc = EIGEN_VECTOR(M_mt)
    xst = np.dot(x_vc, po_vc)
    ueqo, veqo = np.dot(po_vc, u_vc), np.dot(po_vc, v_vc)
    xst_vc.append(xst)
    yst_vc.append(y[0])
    ueqp_vc.append(ueqo)
    veqp_vc.append(veqo)

    # write txt
    txt.write(str(round(t / NdT, 4)) + "\t")
    for l in range(0, 4):
        txt.write(str(x_vc[l]) + "\t")
    txt.write(str(y[0]) + "\t")
    for l in range(0, 4):
        txt.write(str(po_vc[l]) + "\t")
    txt.write(str(ueqo) + "\t" + str(veqo) + "\n")

    dueq_vc = []
    for j in range(0, 4):
        xn_vc = np.copy(x_vc)
        xn_vc[j] += dp
        M_mt = M_MATRIX(xn_vc, y)
        p_vc = EIGEN_VECTOR(M_mt)
        dueq = (np.dot(p_vc, u_vc) - ueqo) / dp
        dueq_vc.append(dueq)
    yn = np.copy(y) + dp
    M_mt = M_MATRIX(x_vc, yn)
    p_vc = EIGEN_VECTOR(M_mt)
    dveq = (np.dot(p_vc, v_vc) - veqo) / dp
    x_vc += x_vc * (1 - x_vc) * dueq_vc / NdT
    y += y * (1 - y) * dveq / NdT
    if t % (NdT * 10) == 0:
        print(t / NdT)

txt.close()
