# discretized multi-memory gradient ascent
# (m,nx,ny)=(2,2,1): m=action number, nx=X's memory length, ny=Y's memory length
# output textfile here as dMMGA_221_test.txt
#


import matplotlib.pyplot as plt
import numpy as np

det = 1  # 1: random initial state, 2: memory-less initial state
if det == 1:
    x_vc, y_vc = np.random.random(16), np.random.random(4)
elif det == 2:
    x_vc, y_vc = 0.6 * np.ones(16), 0.6 * np.ones(4)
xc_vc, yc_vc = np.copy(x_vc), np.copy(y_vc)
uo_vc, vo_vc = [+1, -1, -1, +1], [-1, +1, +1, -1]
u_vc, v_vc = np.outer(uo_vc, np.ones(4)), np.outer(vo_vc, np.ones(4))
u_vc, v_vc = np.reshape(u_vc, [16]), np.reshape(v_vc, [16])
Tmax, NdT = 10**3, 10**3
dp, maxdis = 10**-6, 10**-9


# eigenvector
def EIGEN_VECTOR(M_mt):
    distance = 1
    p_vc = np.ones(16) / 16
    while distance > maxdis:
        pn_vc = np.dot(M_mt, p_vc)
        pn_vc = pn_vc / np.sum(pn_vc)
        distance = np.sum((pn_vc - p_vc) ** 2) ** 0.5
        p_vc = pn_vc
    return p_vc


# matrix generation
def M_MATRIX(x_vc, y_vc):
    four_sixteen_mt = np.outer(np.eye(4), np.ones(4))
    four_sixteen_mt = np.reshape(four_sixteen_mt, [4, 16])
    M1_mt = x_vc * np.reshape(np.outer(y_vc, np.ones(4)), (16)) * four_sixteen_mt
    M2_mt = x_vc * (1 - np.reshape(np.outer(y_vc, np.ones(4)), (16))) * four_sixteen_mt
    M3_mt = (1 - x_vc) * np.reshape(np.outer(y_vc, np.ones(4)), (16)) * four_sixteen_mt
    M4_mt = (
        (1 - x_vc)
        * (1 - np.reshape(np.outer(y_vc, np.ones(4)), (16)))
        * four_sixteen_mt
    )
    M_mt = np.append(M1_mt, M2_mt, axis=0)
    M_mt = np.append(M_mt, M3_mt, axis=0)
    M_mt = np.append(M_mt, M4_mt, axis=0)
    return M_mt


# output textfile
txt = open("dMMGA_221_test.txt", "w")
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
txt.write("#strategies: two-memory vs one-memory" + "\n")
txt.write("#t_vc(1), x_vc(16), y(4), p_vc(16), ust(1), vst(1)" + "\n")

xst_vc, yst_vc = [], []
ueqp_vc, veqp_vc = [], []
for t in range(0, int(Tmax * NdT) + 1):
    M_mt = M_MATRIX(x_vc, y_vc)
    po_vc = EIGEN_VECTOR(M_mt)
    xst = np.dot(x_vc, po_vc)
    yst = np.dot(np.reshape(np.outer(y_vc, np.ones(4)), [16]), po_vc)
    ueqo, veqo = np.dot(po_vc, u_vc), np.dot(po_vc, v_vc)
    xst_vc.append(xst)
    yst_vc.append(yst)
    ueqp_vc.append(ueqo)
    veqp_vc.append(veqo)

    # write txt
    txt.write(str(round(t / NdT, 4)) + "\t")
    for l in range(0, 16):
        txt.write(str(x_vc[l]) + "\t")
    for l in range(0, 4):
        txt.write(str(y_vc[l]) + "\t")
    for l in range(0, 16):
        txt.write(str(po_vc[l]) + "\t")
    txt.write(str(ueqo) + "\t" + str(veqo) + "\n")

    dueq_vc = []
    for j in range(0, 16):
        xn_vc = np.copy(x_vc)
        xn_vc[j] += dp
        M_mt = M_MATRIX(xn_vc, y_vc)
        p_vc = EIGEN_VECTOR(M_mt)
        dueq = (np.dot(p_vc, u_vc) - ueqo) / dp
        dueq_vc.append(dueq)
    dveq_vc = []
    for j in range(0, 4):
        yn_vc = np.copy(y_vc)
        yn_vc[j] += dp
        M_mt = M_MATRIX(x_vc, yn_vc)
        p_vc = EIGEN_VECTOR(M_mt)
        dveq = (np.dot(p_vc, v_vc) - veqo) / dp
        dveq_vc.append(dveq)
    x_vc += x_vc * (1 - x_vc) * dueq_vc / NdT
    y_vc += y_vc * (1 - y_vc) * dveq_vc / NdT
    if t % (NdT * 10) == 0:
        print(t / NdT)

txt.close()
