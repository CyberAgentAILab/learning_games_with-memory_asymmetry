# continualized multi-memory gradient ascent
# 4th-order Runge-Kutta
# forward- and backward-time dynamics
# (m,nx,ny)=(3,1,0): m=action number, nx=X's memory length, ny=Y's memory length
# output textfile here as cMMGA_210_test.txt
#


import numpy as np
import matplotlib.pyplot as plt


NdT = 200
init = 1 # 1: random initial state, 2: neighborhood of Nash equilibrium
if init == 0:
    x_vc, y = np.random.random(4), np.random.random()
elif init == 1:
    scale = 0.1
    x_vc, y = 0.5+scale*np.random.randn(4), 0.5+scale*np.random.randn()
u_vc, v_vc = np.array([+1,-1,-1,+1]), np.array([-1,+1,+1,-1])
dmin = 10**-8


# define function (analytically calculate equilibrium state for fixed strategy)
def SS_ANALYTICAL(x_vc, y):
    [x1, x2, x3, x4] = x_vc
    xst = (x3*y+x4*(1-y))/(1-x1*y-x2*(1-y)+x3*y+x4*(1-y))
    dx1 = x1*(1-x1)*(4*y-2)*xst*y/(1-x1*y-x2*(1-y)+x3*y+x4*(1-y))
    dx2 = x2*(1-x2)*(4*y-2)*xst*(1-y)/(1-x1*y-x2*(1-y)+x3*y+x4*(1-y))
    dx3 = x3*(1-x3)*(4*y-2)*(1-xst)*y/(1-x1*y-x2*(1-y)+x3*y+x4*(1-y))
    dx4 = x4*(1-x4)*(4*y-2)*(1-xst)*(1-y)/(1-x1*y-x2*(1-y)+x3*y+x4*(1-y))
    dy = -y*(1-y)*((4*xst-2)+(4*y-2)*(-(1-x1)*x4+(1-x2)*x3)/(1-x1*y-x2*(1-y)+x3*y+x4*(1-y))**2)
    dx_vc = np.array([dx1, dx2, dx3, dx4])
    return(dx_vc, dy)

def SS_ANALYTICAL_R(x_vc, y):
    [x1, x2, x3, x4] = x_vc
    xst = (x3*y+x4*(1-y))/(1-x1*y-x2*(1-y)+x3*y+x4*(1-y))
    dx1 = x1*(1-x1)*(4*y-2)*xst*y/(1-x1*y-x2*(1-y)+x3*y+x4*(1-y))
    dx2 = x2*(1-x2)*(4*y-2)*xst*(1-y)/(1-x1*y-x2*(1-y)+x3*y+x4*(1-y))
    dx3 = x3*(1-x3)*(4*y-2)*(1-xst)*y/(1-x1*y-x2*(1-y)+x3*y+x4*(1-y))
    dx4 = x4*(1-x4)*(4*y-2)*(1-xst)*(1-y)/(1-x1*y-x2*(1-y)+x3*y+x4*(1-y))
    dy = -y*(1-y)*((4*xst-2)+(4*y-2)*(-(1-x1)*x4+(1-x2)*x3)/(1-x1*y-x2*(1-y)+x3*y+x4*(1-y))**2)
    dx_vc = np.array([dx1, dx2, dx3, dx4])
    return(-dx_vc, -dy)


# output textfile
txt = open('cMMGA_210_test.txt', 'w')
txt.write('#time:[NdT] (Runge-Kutta 4) = ' + str([NdT]) + '\n')
txt.write('#payoff:u = ' + str([u_vc]) + ', v = ' + str([v_vc]) + '\n')
txt.write('#strategies: one-memory vs zero-memory' + '\n')
txt.write('#whole dynamics from source to sink' + '\n')
txt.write('#t, xT_vc(4), yT(1), u(1), v(1)' + '\n')


xst = (x_vc[2]*y+x_vc[3]*(1-y))/(1-x_vc[0]*y-x_vc[1]*(1-y)+x_vc[2]*y+x_vc[3]*(1-y))
while ((xst-0.5)**2+(y-0.5)**2)**0.5 > dmin or (1-x_vc[0])*x_vc[3]-(1-x_vc[1])*x_vc[2] > 0:
    # analytical
    # Runge-Kutta k
    dxk_vc, dyk = SS_ANALYTICAL(x_vc, y)
    xk_vc, yk = x_vc+dxk_vc/NdT/2, y+dyk/NdT/2
    # Runge-Kutta l
    dxl_vc, dyl = SS_ANALYTICAL(xk_vc, yk)
    xl_vc, yl = x_vc+dxl_vc/NdT/2, y+dyl/NdT/2
    # Runge-Kutta m
    dxm_vc, dym = SS_ANALYTICAL(xl_vc, yl)
    xm_vc, ym = x_vc+dxm_vc/NdT, y+dym/NdT
    # Runge-Kutta n
    dxn_vc, dyn = SS_ANALYTICAL(xm_vc, ym)
    dx_vc, dy = (dxk_vc+2*dxl_vc+2*dxm_vc+dxn_vc)/6, (dyk+2*dyl+2*dym+dyn)/6

    x_vc += dx_vc/NdT
    y += dy/NdT
    xst = (x_vc[2]*y+x_vc[3]*(1-y))/(1-x_vc[0]*y-x_vc[1]*(1-y)+x_vc[2]*y+x_vc[3]*(1-y))

xf_vc = np.copy(x_vc)

while ((xst-0.5)**2+(y-0.5)**2)**0.5 >  dmin or (1-x_vc[0])*x_vc[3]-(1-x_vc[1])*x_vc[2] < 0:
    # analytical
    # Runge-Kutta k
    dxk_vc, dyk = SS_ANALYTICAL_R(x_vc, y)
    xk_vc, yk = x_vc+dxk_vc/NdT/2, y+dyk/NdT/2
    # Runge-Kutta l
    dxl_vc, dyl = SS_ANALYTICAL_R(xk_vc, yk)
    xl_vc, yl = x_vc+dxl_vc/NdT/2, y+dyl/NdT/2
    # Runge-Kutta m
    dxm_vc, dym = SS_ANALYTICAL_R(xl_vc, yl)
    xm_vc, ym = x_vc+dxm_vc/NdT, y+dym/NdT
    # Runge-Kutta n
    dxn_vc, dyn = SS_ANALYTICAL_R(xm_vc, ym)
    dx_vc, dy = (dxk_vc+2*dxl_vc+2*dxm_vc+dxn_vc)/6, (dyk+2*dyl+2*dym+dyn)/6

    x_vc += dx_vc/NdT
    y += dy/NdT
    xst = (x_vc[2]*y+x_vc[3]*(1-y))/(1-x_vc[0]*y-x_vc[1]*(1-y)+x_vc[2]*y+x_vc[3]*(1-y))


t = 0.0
x1_nor_vc, x2_nor_vc, x3_nor_vc, x4_nor_vc = [], [], [], []
xst_nor_vc, y_nor_vc = [], []
while ((xst-0.5)**2+(y-0.5)**2)**0.5 > dmin or (1-x_vc[0])*x_vc[3]-(1-x_vc[1])*x_vc[2] > 0:
    xst = (x_vc[2]*y+x_vc[3]*(1-y))/(1-x_vc[0]*y-x_vc[1]*(1-y)+x_vc[2]*y+x_vc[3]*(1-y))
    pst_vc = np.reshape([xst*y, xst*(1-y), (1-xst)*y, (1-xst)*(1-y)], 4)

    # write txt
    txt.write(str(round(t,3))+'\t')
    for l in range(0,4):
        txt.write(str(x_vc[l]-0.5)+'\t')
    txt.write(str(xst-0.5)+'\t')
    txt.write(str(y-0.5)+'\t')
    txt.write(str(np.dot(pst_vc,u_vc))+'\t'+str(np.dot(pst_vc,v_vc))+'\n')

    x1_nor_vc.append(x_vc[0])
    x2_nor_vc.append(x_vc[1])
    x3_nor_vc.append(x_vc[2])
    x4_nor_vc.append(x_vc[3])
    xst_nor_vc.append(xst)
    y_nor_vc.append(y)

    # analytical
    # Runge-Kutta k
    dxk_vc, dyk = SS_ANALYTICAL(x_vc, y)
    xk_vc, yk = x_vc+dxk_vc/NdT/2, y+dyk/NdT/2
    # Runge-Kutta l
    dxl_vc, dyl = SS_ANALYTICAL(xk_vc, yk)
    xl_vc, yl = x_vc+dxl_vc/NdT/2, y+dyl/NdT/2
    # Runge-Kutta m
    dxm_vc, dym = SS_ANALYTICAL(xl_vc, yl)
    xm_vc, ym = x_vc+dxm_vc/NdT, y+dym/NdT
    # Runge-Kutta n
    dxn_vc, dyn = SS_ANALYTICAL(xm_vc, ym)
    dx_vc, dy = (dxk_vc+2*dxl_vc+2*dxm_vc+dxn_vc)/6, (dyk+2*dyl+2*dym+dyn)/6

    x_vc += dx_vc/NdT
    y += dy/NdT
    t += 1.0/NdT
    xst = (x_vc[2]*y+x_vc[3]*(1-y))/(1-x_vc[0]*y-x_vc[1]*(1-y)+x_vc[2]*y+x_vc[3]*(1-y))
    

# write txt
txt.write(str(round(t,3))+'\t')
for l in range(0,4):
    txt.write(str(x_vc[l]-0.5)+'\t')
txt.write(str(xst-0.5)+'\t')
txt.write(str(y-0.5)+'\t')
txt.write(str(np.dot(pst_vc,u_vc))+'\t'+str(np.dot(pst_vc,v_vc))+'\n')
txt.close()

if np.sum((x_vc-xf_vc)**2)**0.5 < dmin:
    print('consistent!')




