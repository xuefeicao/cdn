from __future__ import print_function
import numpy as np
import multiprocessing as mp
import time
from scipy.integrate import simps
from functools import partial
from cdn.validation_truncation_1 import cross_validation
from cdn.model_config import Modelconfig, Modelpara
import os
from six.moves import cPickle as pkl
import random
import glob
import six 
def error_ws_0(y, gamma_ini, lam_1, P12, Omega):
    n_area = y.shape[0]
    e1 = np.sum((y-np.dot(gamma_ini,np.transpose(P12)))**2)
    plt_1 = 0
    for i in range(n_area):
        plt_1 = plt_1 + np.dot(np.dot(gamma_ini[i,:],Omega),gamma_ini[i,:])
    return e1+lam_1*plt_1

def error_ws(y, gamma_ini, lam_1, P12, Omega):
    stp=1
    while(stp<1000):
        gr=np.dot((np.dot(gamma_ini,np.transpose(P12))-y),P12)*2+2*lam_1*np.dot(gamma_ini,np.transpose(Omega))
        n_gr=(np.sum(gr**2))
        f_t=1
        fixed=error_ws_0(y, gamma_ini, lam_1, P12, Omega)
        while(error_ws_0(y, gamma_ini-f_t*gr, lam_1, P12, Omega)>fixed-0.5*f_t*n_gr):
            f_t=0.8*f_t
        gamma_ini=gamma_ini-gr*f_t
        stp=stp+1
        if n_gr**0.5<0.001:
            break
    return gamma_ini
def update_p(file_name_dir, precomp_dir, pickle_file,  tol, max_iter, multi, lamu):
    """
    The main algorithm, updating parameter for a defined problem

    Parameters
    -----------
    file_name_dir: dir of problem folder
    precomp_dir: dir of precomputed data
    pickele_file: file name which we use to save estimations
    lamu: list = [lam, mu, lam_1], in our paper, mu is set to be zero. lam*mu is the coefficient 
          for l2 norm penalty of A, B, C
    tol, max_iter:
    multi: bool variable, Default True
    """
    configpara = Modelpara(precomp_dir+'precomp.pkl')
    config = Modelconfig(file_name_dir+'data/observed.pkl')
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
    Omega = configpara.Omega
    y = config.y 
    n_area = config.n_area
    p = configpara.p
    t_i = configpara.t_i
    l_t = configpara.l_t
    J = configpara.J 
    t_T = configpara.t_T 
    dt = configpara.dt
    row_n = configpara.row_n
    fold=configpara.fold
    ###################################################################################
    def gr(gamma, A, B, C, D, lam, mu, lam_1):
        g = np.zeros((n_area,p))
        g = g+np.dot(gamma,P1)-np.dot(np.dot(np.transpose(A),gamma),np.transpose(P2))
        g = g-np.dot(np.dot(A,gamma),P2)+np.dot(np.dot(np.dot(np.transpose(A),A),gamma),P5)
        tmp_1 = 0
        tmp_2 = 0 
        for j in range(J):
            tmp_1 = tmp_1+np.dot(np.dot(B[:,:,j],gamma),P3[:,:,j])
            tmp_2 = tmp_2+np.dot(np.dot(np.dot(np.transpose(A),B[:,:,j]),gamma),P6[:,:,j])
        g = g-(tmp_1-tmp_2)
        g = g-np.dot(C,P4)+np.dot(np.dot(np.transpose(A),C),P7)
        g = g-np.dot(D,P8)+np.dot(np.dot(np.transpose(A),D),P9)
        tmp = 0
        for l in range(J):
            tmp_1 = 0
            for j in range(J):
                tmp_1 = np.dot(np.dot(B[:,:,j],gamma),P10[:,:,j,l])
            tmp = tmp-np.dot(np.transpose(B[:,:,l]),(np.dot(gamma,np.transpose(P3[:,:,l]))-np.dot(np.dot(A,gamma),np.transpose(P6[:,:,l]))-tmp_1-np.dot(C,P13[:,:,l])-np.dot(D,P11[l,:].reshape((1,-1)))))
        g += tmp
        g = g*2*lam
        tmp1 = np.zeros((n_area,1))
        tmp2 = np.zeros((n_area,J))

        for m in range(n_area):
            tmp1[m,0] = np.sum(abs(A[:,m]))/np.dot(np.dot(gamma[m,:],P5),gamma[m,])**0.5
            for j in range(J):
                tmp2[m,j] = np.sum(abs(B[:,m,j]))/np.dot(np.dot(gamma[m,:],P10[:,:,j,j]),gamma[m,:])**0.5
        g = g + lam*mu*np.dot(gamma,np.transpose(P5))*tmp1
        for j in range(J):
            g = g + lam*mu*np.dot(gamma,P10[:,:,j,j])*(tmp2[:,j].reshape((-1,1)))
        g = g + np.dot((np.dot(gamma,np.transpose(P12))-y),P12)*2
        g = g + 2*lam_1*np.dot(gamma,np.transpose(Omega))
        g[np.isnan(g)]=0
        return g 
    def cd_thre(tmp, tmp_1, mu):
        mu = mu/2.0
        if abs(tmp)>mu*(tmp_1**0.5):
            return np.sign(tmp)*(abs(tmp)-mu*(tmp_1**0.5))/tmp_1
        else:
            return 0
    def update_A(m, n, gamma, A, B, C, D, mu):
        tmp_0 = 0
        for j in range(J):
            tmp_0 = tmp_0+np.dot(np.dot(np.dot(B[m,:,j],gamma),P6[:,:,j]),gamma[n,:])
        tmp_1 = np.dot(np.dot(gamma[n,:],P5),gamma[n,:])
        tmp = np.dot(np.dot(gamma[n,:],P2),gamma[m,:])-np.dot(np.dot(np.dot(A[m,:],gamma),P5),gamma[n,:])-tmp_0-np.dot(np.dot(C[m,:],P7),gamma[n,:])-D[m,0]*np.dot(gamma[n,:],P9[0,:])+A[m,n]*tmp_1
        return cd_thre(tmp,tmp_1,mu)
    def update_B(m, n, j, gamma, A, B, C, D, mu):
        tmp_0=0
        for l in range(J):
            tmp_0 = tmp_0 + np.dot(np.dot(np.dot(B[m,:,l],gamma),P10[:,:,l,j]),gamma[n,:])
        tmp_1 = np.dot(np.dot(gamma[n,:],P10[:,:,j,j]),gamma[n,:])
        tmp = np.dot(np.dot(gamma[n,:],P3[:,:,j]),gamma[m,:])-np.dot(np.dot(np.dot(A[m,:],gamma),np.transpose(P6[:,:,j])),gamma[n,:])-tmp_0-np.dot(np.dot(C[m,:],P13[:,:,j]),gamma[n,:])-D[m,0]*np.dot(gamma[n,:],P11[j,:])+B[m,n,j]*tmp_1
        return cd_thre(tmp,tmp_1,mu)
    def update_C(m,n,gamma,A,B,C,D,mu):
        tmp_0 = 0
        for j in range(J):
            tmp_0 = tmp_0 + np.dot(np.dot(B[m,:,j],gamma),P13[n,:,j])
        tmp_1=P14[n,n]
        tmp=np.dot(gamma[m,:],P4[n,:])-np.dot(np.dot(A[m,:],gamma),P7[n,:])-tmp_0-np.dot(C[m,:],P14[n,:])-D[m,0]*P15[0,n]+C[m,n]*tmp_1
        return cd_thre(tmp,tmp_1,mu)
    def update_D(gamma, A, B, C):
        tmp = np.dot(gamma,np.transpose(P8))-np.dot(np.dot(A,gamma),np.transpose(P9))
        for j in range(J):
            tmp = tmp - np.dot(np.dot(B[:,:,j],gamma),P11[j,:]).reshape((-1,1))
        tmp = tmp - np.dot(C,np.transpose(P15))
        return tmp*1.0/t_T
    def likelihood(gamma, A, B, C, D, lam, mu, lam_1, p_t=False):
        e1 = np.sum((y-np.dot(gamma,np.transpose(P12)))**2)
        e2 = 0
        tmp_0 = 0
        for j in range(J):
            tmp_0 = tmp_0 + np.dot(np.dot(B[:,:,j],gamma),Q3[:,:,j])
        tmp = np.dot(gamma,Q1)-np.dot(np.dot(A,gamma),Q2)-tmp_0-np.dot(C,Q4)-np.repeat(D,l_t,axis=1) 
        for m in range(n_area):
            e2 = e2 + simps(tmp[m,:]**2,t_i)
        plt = 0
        for k in range(n_area):
            w_1k = np.dot(np.dot(gamma[k,:],P5),gamma[k,:])**0.5
            plt = plt + np.sum(abs(A[:,k]))*w_1k
            for j in range(J):
                w_2kj = np.dot(np.dot(gamma[k,:],P10[:,:,j,j]),gamma[k,:])**0.5
                plt = plt + np.sum(abs(B[:,k,j]))*w_2kj
        for k in range(J):
            w_3k = (P14[k,k])**0.5
            plt = plt + np.sum(abs(C[:,k]))*w_3k
        plt_1 = 0
        for i in range(n_area):
            plt_1 = plt_1 + np.dot(np.dot(gamma[i,:],Omega),gamma[i,:])
        
        sum_e = e1 + lam*e2 + lam*mu*plt + lam_1*plt_1
        if p_t == True:
            return(e1,e2,plt,plt_1)
        return sum_e
    def update_all_3(gamma, mu=0):
        """
        Second step for updating A, B, C

        Parameters
        -----------
        gamma: numpy array, 
        mu : this is an extra tuning parameter which is not used in paper, but provided for people who are interested to add penalty 
        to the l2 norm of A, B, C
        """
        n_all = n_area*(J+1)+J+1
        Y_tmp = np.zeros((n_area,n_all))
        X_tmp = np.zeros((n_all,n_all))
        I_tmp = np.zeros((n_all,n_all))
        W_A = np.zeros((n_area,n_area))
        for i in range(n_area):
             W_A[i,i] = np.dot(np.dot(gamma[i,:],P5),np.transpose(gamma[i,:]))
        I_tmp[0:n_area,0:n_area] = W_A
        W_B=np.zeros((n_area,n_area,J))
        for j in range(J):
            for i in range(n_area):
                W_B[i,i,j] = np.dot(np.dot(gamma[i,:],P10[:,:,j,j]),np.transpose(gamma[i,:]))
            I_tmp[((j+1)*n_area):((j+2)*n_area),((j+1)*n_area):((j+2)*n_area)]=W_B[:,:,j]
        W_C = np.zeros((J,J))
        for j in range(J):
            W_C[j,j] = P14[j,j]
        I_tmp[((J+1)*n_area):((J+1)*n_area+J),((J+1)*n_area):((J+1)*n_area+J)] = W_C
        for j in range(J+1):
            if j == 0:
                Y_tmp[:,j*n_area:((j+1)*n_area)] = np.dot(np.dot(gamma,np.transpose(P2)),np.transpose(gamma))
            else:
                Y_tmp[:,j*n_area:((j+1)*n_area)] = np.dot(np.dot(gamma,np.transpose(P3[:,:,j-1])),np.transpose(gamma))
        Y_tmp[:,((J+1)*n_area):((J+1)*n_area+J)] = np.dot(gamma,np.transpose(P4))
        Y_tmp[:,-1] = np.dot(gamma,np.transpose(P8)).reshape((-1,))
        for j in range(J+1):
            if j == 0:
                X_tmp[j*n_area:((j+1)*n_area),0:n_area] = np.dot(np.dot(gamma,P5),np.transpose(gamma))
            else:
                X_tmp[j*n_area:((j+1)*n_area),0:n_area] = np.dot(np.dot(gamma,P6[:,:,j-1]),np.transpose(gamma))
        X_tmp[((J+1)*n_area):((J+1)*n_area+J),0:n_area] = np.dot(P7,np.transpose(gamma))
        X_tmp[-1,0:n_area] = np.dot(P9,np.transpose(gamma))
        tmp = np.zeros((n_area*J,n_area*J))
        for j in range(J):
            for l in range(J):
                tmp[j*n_area:(j+1)*n_area,l*n_area:(l+1)*n_area] = np.dot(np.dot(gamma,P10[:,:,j,l]),np.transpose(gamma))
        for j in range(J):
            X_tmp[0:n_area,((j+1)*n_area):((j+2)*n_area)] = np.dot(np.dot(gamma,np.transpose(P6[:,:,j])),np.transpose(gamma))
            X_tmp[(J+1)*n_area:(J+1)*n_area+J,((j+1)*n_area):((j+2)*n_area)] = np.dot(P13[:,:,j],np.transpose(gamma))
            X_tmp[-1,((j+1)*n_area):((j+2)*n_area)] = np.dot(P11[j,:].reshape((1,-1)),np.transpose(gamma))
        X_tmp[n_area:(J+1)*n_area,n_area:(J+1)*n_area] = tmp
        ##C
        X_tmp[0:n_area,(J+1)*n_area:((J+1)*n_area+J)] = np.dot(gamma,np.transpose(P7))
        for j in range(J):
            X_tmp[n_area*(j+1):n_area*(j+2),(J+1)*n_area:((J+1)*n_area+J)] = np.dot(gamma,np.transpose(P13[:,:,j]))
        X_tmp[(J+1)*n_area:((J+1)*n_area+J),(J+1)*n_area:((J+1)*n_area+J)] = P14
        X_tmp[-1,(J+1)*n_area:((J+1)*n_area+J)]=P15
        ##D
        X_tmp[0:n_area,-1] = np.dot(gamma,np.transpose(P9)).reshape((-1))
        for j in range(J):
            X_tmp[n_area*(j+1):n_area*(j+2),-1] = np.dot(gamma,np.transpose(P11[j,:])).reshape((-1))
        X_tmp[(J+1)*n_area:((J+1)*n_area+J),-1] = np.transpose(P15).reshape((-1))
        X_tmp[-1,-1] = t_T 
        #s_eig = np.sort(abs(np.linalg.eig(X_tmp)[0]))
        #print(np.linalg.cond(X_tmp+mu*I_tmp), s_eig[-1] ,s_eig[0])
        if config.D_u == False:
            Y_tmp = Y_tmp[:,0:-1]
            X_tmp = X_tmp[0:-1,0:-1]
            I_tmp = I_tmp[0:-1,0:-1]
            return np.dot(Y_tmp,np.linalg.pinv(X_tmp+mu*I_tmp))
        return np.dot(Y_tmp,np.linalg.pinv(X_tmp+mu*I_tmp))

    def update_all_2(gamma, mu):
        """
        For the case when B = 0
        """
        n_all = n_area+J+1
        Y_tmp = np.zeros((n_area,n_all))
        X_tmp = np.zeros((n_all,n_all))
        I_tmp = np.zeros((n_all,n_all))
        W_A = np.zeros((n_area,n_area))
        for i in range(n_area):
             W_A[i,i] = np.dot(np.dot(gamma[i,:],P5),np.transpose(gamma[i,:]))
        I_tmp[0:n_area,0:n_area] = W_A
        W_C = np.zeros((J,J))
        for j in range(J):
            W_C[j,j] = P14[j,j]
        I_tmp[((1)*n_area):((1)*n_area+J),((1)*n_area):((1)*n_area+J)] = W_C
        Y_tmp[:,0:n_area] = np.dot(np.dot(gamma,np.transpose(P2)),np.transpose(gamma))
        Y_tmp[:,((1)*n_area):((1)*n_area+J)] = np.dot(gamma,np.transpose(P4))
        Y_tmp[:,-1] = np.dot(gamma,np.transpose(P8)).reshape((-1,))
        X_tmp[0:n_area,0:n_area] = np.dot(np.dot(gamma,P5),np.transpose(gamma))
        X_tmp[((1)*n_area):((1)*n_area+J),0:n_area] = np.dot(P7,np.transpose(gamma))
        X_tmp[-1,0:n_area] = np.dot(P9,np.transpose(gamma))
        ##C
        X_tmp[0:n_area,n_area:(n_area+J)] = np.dot(gamma,np.transpose(P7))
        X_tmp[n_area:(n_area+J),n_area:(n_area+J)] = P14
        X_tmp[-1, n_area: (n_area+J)] = P15
        ##D
        X_tmp[0:n_area,-1] = np.dot(gamma,np.transpose(P9)).reshape((-1))
        X_tmp[n_area:(n_area+J),-1] = np.transpose(P15).reshape((-1))
        X_tmp[-1,-1] = t_T 
        #s_eig = np.sort(abs(np.linalg.eig(X_tmp)[0]))
        if config.D_u == False:
            Y_tmp = Y_tmp[:,0:-1]
            X_tmp = X_tmp[0:-1,0:-1]
            I_tmp = I_tmp[0:-1,0:-1]
            return np.dot(Y_tmp,np.linalg.pinv(X_tmp+mu*I_tmp))
        return np.dot(Y_tmp,np.linalg.pinv(X_tmp+mu*I_tmp))

    def update_all_1(gamma, mu):
        """
        For the case B = 0 and C = 0
        """

        n_all = n_area + 1
        Y_tmp = np.zeros((n_area,n_all))
        X_tmp = np.zeros((n_all,n_all))
        I_tmp = np.zeros((n_all,n_all))
        W_A = np.zeros((n_area,n_area))
        for i in range(n_area):
             W_A[i,i] = np.dot(np.dot(gamma[i,:],P5),np.transpose(gamma[i,:]))
        I_tmp[0:n_area,0:n_area] = W_A
  
        Y_tmp[:,0:n_area] = np.dot(np.dot(gamma,np.transpose(P2)),np.transpose(gamma))
  
        Y_tmp[:,-1] = np.dot(gamma,np.transpose(P8)).reshape((-1,))


        X_tmp[0:n_area,0:n_area] = np.dot(np.dot(gamma,P5),np.transpose(gamma))
      
        X_tmp[-1,0:n_area] = np.dot(P9,np.transpose(gamma))
        ##D
        X_tmp[0:n_area,-1] = np.dot(gamma,np.transpose(P9)).reshape((-1))
        X_tmp[-1,-1] = t_T 
        s_eig = np.sort(abs(np.linalg.eig(X_tmp)[0]))
        #print(np.linalg.cond(X_tmp+mu*I_tmp), s_eig[-1] ,s_eig[0])
        if config.D_u == False:
            Y_tmp = Y_tmp[:,0:-1]
            X_tmp = X_tmp[0:-1,0:-1]
            I_tmp = I_tmp[0:-1,0:-1]
            return np.dot(Y_tmp,np.linalg.pinv(X_tmp+mu*I_tmp))
        return np.dot(Y_tmp,np.linalg.pinv(X_tmp+mu*I_tmp))


    #######################################################################################
    ##############################################################################################

    def ini_select(y, lam_1, P12=P12, Omega=Omega):
        """
        selecting an initial for gamma which may help to avoid local minimum

        Parameters
        ------------- 
        lam_1: scalar, penalty for the second derivative of neuronal activities x. 
        """
        gamma_0 = np.zeros((n_area,p))
        gamma_0 = error_ws(y, gamma_0, lam_1, P12, Omega)
        return gamma_0 

    def str_1(num):
        if num>=1:
            return str(int(num))
        num=str(num)
        num_1=''
        for i in range(len(num)):
            if num[i]!='.':
                num_1=num_1+num[i]
        return num_1


    ############################################################################################
    lam = lamu[0]
    mu = lamu[1]
    lam_1=lamu[-1]
    A = np.zeros((n_area,n_area))
    B = np.zeros((n_area,n_area,J))
    C = np.zeros((n_area,J))
    D = np.zeros((n_area,1))
    iter = 0
    sum_e = 10**6
    gamma = ini_select(y, lam_1) 
    sum_e_1 = likelihood(gamma, A, B, C, D, lam, mu, lam_1, p_t=True)[1]

    while(iter<max_iter and abs(sum_e-sum_e_1)/sum_e_1>tol):
        #gamma_1 = gamma + np.ones((n_area,p))
        stp = 1
        while(stp<10 and iter>1):
            results = gr(gamma,A,B,C,D,lam,mu,lam_1)
            n_results = (np.sum(results**2))
            gamma_1 = gamma.copy()
            f_t = 1
            fixed = likelihood(gamma,A,B,C,D,lam,mu,lam_1)
            while likelihood(gamma-f_t*results,A,B,C,D,lam,mu,lam_1)>fixed-0.5*f_t*n_results:
                f_t = 0.8*f_t
            gamma = gamma-results*f_t
            stp = stp+1

        if config.B_u == True:
            tmp = update_all_3(gamma,mu=0)
            A = tmp[:,0:n_area]
            for j in range(J):
                B[:,:,j] = tmp[:,((j+1)*n_area):((j+2)*n_area)]
            C = tmp[:,((J+1)*n_area):((J+1)*n_area+J)]
            if config.D_u == True:
                D = tmp[:,-1].reshape((-1,1))
        elif config.C_u == True:
            tmp = update_all_2(gamma,mu=0)
            A = tmp[:,0:n_area]
            C = tmp[:,n_area:(n_area+J)]
            if config.D_u == True:
                D = tmp[:,-1].reshape((-1,1))
        else:
            tmp = update_all_1(gamma,mu=0)
            A = tmp[:,0:n_area]
            if config.D_u == True:
                D = tmp[:,-1].reshape((-1,1))
        sum_e = sum_e_1
        sum_e_1 = likelihood(gamma,A,B,C,D,lam,mu,lam_1,p_t=True)[1]
        iter += 1
    e1,e2,plt,plt_1 = likelihood(gamma,A,B,C,D,lam,mu,lam_1,p_t=True)

    if multi == False:
        config.gamma = gamma
        config.A = A
        config.B = B
        config.C = C
        config.D = D
        config.lamu = lamu
        config.e1 = e1
        config.e2 = e2
        config.plt = plt 
        config.plt_1 = plt_1
        config.t_i = configpara.t_i
        pickle_file_1 = file_name_dir + 'results/result.pkl'
        f = open(pickle_file_1, 'wb')
        save = {
        'estimated_x': np.dot(config.gamma,configpara.Q2_all),
        'y': config.y, 
        'estimated_y': np.dot(config.gamma,np.transpose(P12)), 
        #'x_real': config.x_real, 
        'gamma': config.gamma,
        'A': config.A,
        'B': config.B,
        'C': config.C,
        'D':config.D,
        #'A_real': config.A_real,
        #'B_real': config.B_real,
        #'C_real': config.C_real,
        #'D_real': config.D_real,
        'lamu': config.lamu, 
        'e1': config.e1, 'e2': config.e2, 'plt_1': config.plt_1, 'plt': config.plt,
        't': np.arange(0,configpara.dt*(configpara.row_n-1)+configpara.dt**0.5,configpara.dt),
        'n1': (int(configpara.t_i[0]/configpara.dt)+1) #valid estimation bound
        }
        pkl.dump(save, f, pkl.HIGHEST_PROTOCOL)
        f.close()
        return
    else:
        pickle_file_1 = pickle_file + str_1(lam)+'_'+str_1(mu*lam) + '_' + str_1(lam_1)+'.pickle'
        f = open(pickle_file_1, 'wb')
        save = {
        'result': [lamu,gamma,A,B,C,D,e1,e2,plt,plt_1]
        }
        pkl.dump(save, f, pkl.HIGHEST_PROTOCOL)
        f.close()
        return 
def str_2(num):
    if num[0] == '0':
        return float(num)/(10**(len(num)-1))
    else:
        return float(num)
    
def select_lamu(lam, mu, lam_1, file_name_dir, pickle_file, precomp_dir, val_data_dir=None, val_precomp_dir=None, num_cores=1, tol=1e-2, max_iter=100):
    """
    wrapper for selecting the tuning parameters of one subject
    See function update_p for details of variables meaning

    Parameters
    -----------
    num_cores : int, allow multi-processing, default None

    Returns
    -----------
    An instance of Modelconfig, including all summaries of estimation for one subject
    """
    para = list()
   
    for i in range(len(lam)):
        for j in range(len(mu)):
            for k in range(len(lam_1)):
                para.append((lam[i], mu[j], lam_1[k]))
    if len(para) >= 1:
        if num_cores > 1:
            pool = mp.Pool(processes=min(len(para), num_cores))
            print('begin multiprocessing with {0} cores'.format(num_cores))
            update_p_1 = partial(update_p, file_name_dir, precomp_dir, pickle_file, tol, max_iter, True)
            pool.map(update_p_1,para)
            pool.close()
            pool.join()
        else:
            for i in range(len(para)):
                update_p(file_name_dir, precomp_dir, pickle_file, tol, max_iter, True, para[i])
    results = list()
    file_config = glob.glob(pickle_file+'*.pickle')
    for i in range(len(file_config)):
        f = open(file_config[i], 'rb')
        if six.PY2:
            save = pkl.load(f)
        else:
            save = pkl.load(f, encoding='latin1')
        results.append(save['result'])
    pickle_file_1 = file_name_dir + 'results/result.pkl'
    config = Modelconfig(file_name_dir+'data/observed.pkl')

    
    if not val_data_dir or not val_precomp_dir:
        val_data_dir = precomp_dir
        val_precomp_dir = precomp_dir


    configpara = Modelpara(val_precomp_dir + 'precomp.pkl')
    with open(val_data_dir + 'observed.pkl', 'rb') as f:
        if six.PY2:
            y = pkl.load(f)['y']
        else:
            y = pkl.load(f, encoding='latin1')['y']

    if len(results) > 1:
        ind, _ = cross_validation(y, configpara, results)
    else:
        ind = 0

    
    config.t_i = configpara.t_i
    config.lamu = results[ind][0]
    config.A = results[ind][2]
    config.B = results[ind][3]
    config.C = results[ind][4]
    config.D = results[ind][5]
    config.gamma = results[ind][1]
    config.e1 = results[ind][6]
    config.e2 = results[ind][7]
    config.plt = results[ind][8]
    config.plt_1 = results[ind][9]
    Q2 = configpara.Q2_all 
    fold = configpara.fold 
    
    f = open(pickle_file_1, 'wb')
    save = {
    'estimated_x': np.dot(config.gamma, Q2[:,0:(Q2.shape[1]+1):int(1/fold)]),
    'y': config.y, 
    'estimated_y': np.dot(config.gamma,np.transpose(configpara.P12)), 
    #'x_real': config.x_real, 
    'gamma': config.gamma,
    'A': config.A,
    'B': config.B,
    'C': config.C,
    'D':config.D,
    #'A_real': config.A_real,
    #'B_real': config.B_real,
    #'C_real': config.C_real,
    #'D_real': config.D_real,
    'lamu': config.lamu, 
    'e1': config.e1, 'e2': config.e2, 'plt_1': config.plt_1, 'plt': config.plt,
    't': np.arange(0,configpara.dt*(configpara.row_n-1)+configpara.dt*0.5,configpara.dt),
    'n1': (int(configpara.t_i[0]/configpara.dt)+1) #valid estimation bound
    }
    pkl.dump(save, f, pkl.HIGHEST_PROTOCOL)
    f.close()

    return config





   

