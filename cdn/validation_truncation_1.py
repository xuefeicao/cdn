from __future__ import print_function
import numpy as np
from scipy.integrate import simps
import math


def cross_validation(y, configpara, results):
    """
    select the tuning parameters with validation 
    """
    P1 = configpara.P1 
    P2 = configpara.P2
    P3 = configpara.P3
    P4 = configpara.P4
    P5 = configpara.P5
    P6 = configpara.P6
    P7 = configpara.P7 
    P8 = configpara.P8
    P9 = configpara.P9
    P10 = configpara.P10 
    P11 = configpara.P11
    P12 = configpara.P12
    P13 = configpara.P13
    P14 = configpara.P14
    P15 = configpara.P15
    Q1 = configpara.Q1
    Q2 = configpara.Q2
    Q3 = configpara.Q3
    Q4 = configpara.Q4
    Q1_1 = configpara.Q1_all
    Q2_1 = configpara.Q2_all
    Q3_1 = configpara.Q3_all
    Q4_1 = configpara.Q4_all 
    n_area = y.shape[0]
    p = configpara.p
    t_i_all = configpara.t_i_all
    l_t_all = configpara.l_t_all
    t_i = configpara.t_i
    l_t = configpara.l_t
    J = configpara.J 
    l_t_0 = configpara.row_n
    l_t_1 = configpara.l_t_1 
    fold = configpara.fold
    t_1 = configpara.t_1
    hrf = configpara.hrf
    t_U = configpara.t_U
    t_U_1 = configpara.t_U_all
    n1 = math.floor(t_i[0]/configpara.dt)+1
    row_n = configpara.row_n

    def ode_x(A, B, C, D1, x_pre):
        h = fold*configpara.dt
        x_l = np.zeros((n_area,l_t_all))
        x_l[:,0] = x_pre[:,0]
        for i in range(1,l_t_all):
            base = x_l[:,i-1]
            if (i-1)%int(1/fold) == 0:
                base = x_pre[:,i-1]
            tmp = 0
            for j in range(J):
                tmp = tmp+Q4_1[j,i-1]*np.dot(B[:,:,j],base)
            k1 =np.dot(A,base) + tmp + np.dot(C,Q4_1[:,i-1])+D1.reshape((-1,))
    
            tmp = 0
            for j in range(J):
                tmp = tmp + t_U_1[j,i-1]*np.dot(B[:,:,j],(base+h/2*k1))
            k2 = np.dot(A,(base+h/2*k1))+ tmp+ np.dot(C,t_U_1[:,i-1])+D1.reshape((-1,))
            tmp = 0
            for j in range(J):
                tmp = tmp + t_U_1[j,i-1]*np.dot(B[:,:,j],(base+h/2*k2))
            k3 = np.dot(A,(base+h/2*k2))+ tmp+ np.dot(C,t_U_1[:,i-1])+D1.reshape((-1,))
            tmp = 0
            for j in range(J):
                tmp = tmp + Q4_1[j,i]*np.dot(B[:,:,j],(base+h*k3))
            k4 = np.dot(A,(base+h*k3))+ tmp + np.dot(C,Q4_1[:,i])+D1.reshape((-1,))
            x_l[:,i] = base+1.0*h/6*(k1+2*k2+2*k3+k4)
        return x_l

    def error_1(A, B, C, D, x_pre):
        x_l = ode_x(A,B,C,D,x_pre)
        z = np.zeros((n_area,l_t_0))
        for j in range(l_t_0):
            tmp = np.zeros((n_area,l_t_1))
            j_1 = int(1/fold)*j+1
            in_1 = min(j_1,l_t_1)

            if j_1-in_1-1 >= 0:
                tmp[:,0:in_1] = x_l[:,(j_1-1):(j_1-1-in_1):-1]
            else:
                tmp[:,0:in_1] = x_l[:,(j_1-1)::-1]

            for m in range(n_area):
                z[m,j] = simps(tmp[m,:]*hrf,t_1)
        e1 = np.sum((y-z)**2)
        return e1,x_l[:,::int(1/fold)]

    E1=list()
    X=list()
    for i in range(len(results)):
        A=results[i][2]
        B=results[i][3]
        C=results[i][4]
        D=results[i][5]
        x_pre=np.dot(results[i][1],Q2_1)
        e1,x=error_1(A,B,C,D,x_pre)

        X.append(x)
        
        E1.append(e1)
 
        
    ind=np.argsort(E1)[0]
    print('selected tuning para:', results[ind][0])
    r_ind=ind
    return r_ind,X[r_ind]

