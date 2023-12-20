"""
discretized multi-memory gradient ascent
(m,nx,ny)=(4,1,0): m=action number, nx=X's memory length, ny=Y's memory length
output textfile here as dMMGA_410_test.txt
"""

import numpy as np

det = 1  # 1: random initial state, 2: memory-less initial state
if det == 1:
    RN_vc = np.sort(np.random.random([17, 3]))
    x_vc = RN_vc[:16, 0]
    x_vc = np.append(x_vc, RN_vc[:16, 1] - RN_vc[:16, 0])
    x_vc = np.append(x_vc, RN_vc[:16, 2] - RN_vc[:16, 1])
    x_vc = np.append(x_vc, 1 - RN_vc[:16, 2])
    x_vc = np.reshape(x_vc, (4, 16))
    x_vc = np.transpose(x_vc)
    x_vc = np.reshape(x_vc, (64))
    y_vc = [
        RN_vc[16, 0],
        RN_vc[16, 1] - RN_vc[16, 0],
        RN_vc[16, 2] - RN_vc[16, 1],
        1 - RN_vc[16, 2],
    ]
elif det == 2:
    x_vc = np.outer(np.ones(16), [0.7, 0.1, 0.1, 0.1])
    x_vc = np.reshape(x_vc, (64))
    y_vc = np.array([0.1, 0.1, 0.7, 0.1])
x_vc, y_vc = np.array(x_vc), np.array(y_vc)
xc_vc, yc_vc = np.copy(x_vc), np.copy(y_vc)
uo_vc, vo_vc = [0, -1, +1, 0, 0, 0, -1, +1, +1, 0, 0, -1, -1, +1, 0, 0], [
    0,
    +1,
    -1,
    0,
    0,
    0,
    +1,
    -1,
    -1,
    0,
    0,
    +1,
    +1,
    -1,
    0,
    0,
]
u_vc, v_vc = np.array(uo_vc) * 1.0, np.array(vo_vc) * 1.0
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
    M_mt = np.array([])
    for j in range(0, 16):
        m_vc = np.reshape(np.outer(x_vc[4 * j : 4 * (j + 1)], y_vc), (16))
        M_mt = np.append(M_mt, m_vc)
    M_mt = np.transpose(np.reshape(M_mt, (16, 16)))
    return M_mt

def main(x_vc, y_vc):
    # output textfile
    txt = open("dMMGA_410_test.txt", "w")
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
    txt.write("#payoff:u = " + str([uo_vc]) + "\n")
    txt.write("#payoff:v = " + str([vo_vc]) + "\n")
    txt.write("#strategies: one-memory vs zero-memory" + "\n")
    txt.write("#t_vc(1), x_vc(64), y(4), p_vc(16), ust(1), vst(1)" + "\n")
    
    xst1_vc, xst2_vc, xst3_vc, xst4_vc = [], [], [], []
    yst1_vc, yst2_vc, yst3_vc, yst4_vc = [], [], [], []
    ueqp_vc, veqp_vc = [], []
    for t in range(0, int(Tmax * NdT) + 1):
        M_mt = M_MATRIX(x_vc, y_vc)
        po_vc = EIGEN_VECTOR(M_mt)
        xst1, xst2, xst3, xst4 = (
            np.dot(x_vc[::4], po_vc),
            np.dot(x_vc[1::4], po_vc),
            np.dot(x_vc[2::4], po_vc),
            np.dot(x_vc[3::4], po_vc),
        )
        yst1, yst2, yst3, yst4 = y_vc[0], y_vc[1], y_vc[2], y_vc[3]
        ueqo, veqo = np.dot(po_vc, u_vc), np.dot(po_vc, v_vc)
        xst1_vc.append(xst1)
        xst2_vc.append(xst2)
        xst3_vc.append(xst3)
        xst4_vc.append(xst4)
        yst1_vc.append(yst1)
        yst2_vc.append(yst2)
        yst3_vc.append(yst3)
        yst4_vc.append(yst4)
        ueqp_vc.append(ueqo)
        veqp_vc.append(veqo)
    
        # write txt
        txt.write(str(round(t / NdT, 4)) + "\t")
        for l in range(0, 64):
            txt.write(str(x_vc[l]) + "\t")
        for l in range(0, 4):
            txt.write(str(y_vc[l]) + "\t")
        for l in range(0, 16):
            txt.write(str(po_vc[l]) + "\t")
        txt.write(str(ueqo) + "\t" + str(veqo) + "\n")
    
        dueq_vc = []
        for j in range(0, 64):
            xn_vc = np.copy(x_vc)
            xn_vc[j] += dp
            jmod4 = int(j / 4)
            xn_vc[4 * jmod4 : 4 * (jmod4 + 1)] = xn_vc[
                4 * jmod4 : 4 * (jmod4 + 1)
            ] / np.sum(xn_vc[4 * jmod4 : 4 * (jmod4 + 1)])
            M_mt = M_MATRIX(xn_vc, y_vc)
            p_vc = EIGEN_VECTOR(M_mt)
            dueq = (np.dot(p_vc, u_vc) - ueqo) / dp
            dueq_vc.append(dueq)
        dveq_vc = []
        for j in range(0, 4):
            yn_vc = np.copy(y_vc)
            yn_vc[j] += dp
            yn_vc = yn_vc / np.sum(yn_vc)
            M_mt = M_MATRIX(x_vc, yn_vc)
            p_vc = EIGEN_VECTOR(M_mt)
            dveq = (np.dot(p_vc, v_vc) - veqo) / dp
            dveq_vc.append(dveq)
        x_vc += x_vc * dueq_vc / NdT
        y_vc += y_vc * dveq_vc / NdT
        x_vc = np.transpose(np.reshape(x_vc, (16, 4)))
        x_vc = x_vc / np.sum(x_vc, axis=0)
        x_vc = np.reshape(np.transpose(x_vc), (64))
        y_vc = y_vc / np.sum(y_vc)
        if t % (NdT * 10) == 0:
            print(t / NdT)
    
    txt.close()

if __name__ == "__main__":
    main(x_vc, y_vc)
