import numpy as np
from scipy.integrate import simps
import math
def cross_validation(config,configpara,results):
    P1=configpara.P1 
    P2=configpara.P2
    P3=configpara.P3
    P4=configpara.P4
    P5=configpara.P5
    P6=configpara.P6
    P7=configpara.P7 
    P8=configpara.P8
    P9=configpara.P9
    P10=configpara.P10 
    P11=configpara.P11
    P12=configpara.P12
    P13=configpara.P13
    P14=configpara.P14
    P15=configpara.P15
    Q1=configpara.Q1
    Q2=configpara.Q2
    Q3=configpara.Q3
    Q4=configpara.Q4
    Q1_1=configpara.Q1_all
    Q2_1=configpara.Q2_all
    Q3_1=configpara.Q3_all
    Q4_1=configpara.Q4_all
    y=config.y 
    n_area=config.n_area
    p=configpara.p
    t_i_all=configpara.t_i_all
    l_t_all=configpara.l_t_all
    t_i=configpara.t_i
    l_t=configpara.l_t
    J=configpara.J 
    l_t_0=configpara.row_n
    l_t_1=configpara.l_t_1 
    fold=configpara.fold
    t_1=configpara.t_1
    hrf=configpara.hrf
    t_U=configpara.t_U
    t_U_1=configpara.t_U_all
    n1=math.floor(t_i[0]/configpara.dt)+1
    row_n=configpara.row_n
    def ode_x(A,B,C,D1,gamma):
        h=fold*configpara.dt
        x_l=np.zeros((n_area,l_t_all))
        x_l[:,0]=np.dot(gamma,Q2_1)[:,0]
        for i in range(1,l_t_all):
            tmp=0

            for j in range(J):
                tmp=tmp+Q4_1[j,i-1]*np.dot(B[:,:,j],x_l[:,i-1])
            k1=np.dot(A,x_l[:,i-1]) + tmp + np.dot(C,Q4_1[:,i-1])+D1.reshape((-1,))
    
            tmp=0
            for j in range(J):
                tmp=tmp+t_U_1[j,i-1]*np.dot(B[:,:,j],(x_l[:,i-1]+h/2*k1))
            k2=np.dot(A,(x_l[:,i-1]+h/2*k1))+ tmp+ np.dot(C,t_U_1[:,i-1])+D1.reshape((-1,))
            tmp=0
            for j in range(J):
                tmp=tmp+t_U_1[j,i-1]*np.dot(B[:,:,j],(x_l[:,i-1]+h/2*k2))
            k3=np.dot(A,(x_l[:,i-1]+h/2*k2))+ tmp+ np.dot(C,t_U_1[:,i-1])+D1.reshape((-1,))
            tmp=0
            for j in range(J):
                tmp=tmp+Q4_1[j,i]*np.dot(B[:,:,j],(x_l[:,i-1]+h*k3))
            k4=np.dot(A,(x_l[:,i-1]+h*k3))+ tmp+ np.dot(C,Q4_1[:,i])+D1.reshape((-1,))
            x_l[:,i]=x_l[:,i-1]+1.0*h/6*(k1+2*k2+2*k3+k4)
        return x_l

    def error_1(A,B,C,D,gamma):
        x_l=ode_x(A,B,C,D,gamma)
        z=np.zeros((n_area,l_t_0))
        for j in range(l_t_0):
            tmp=np.zeros((n_area,l_t_1))
            j_1=int(1/fold)*j+1
            in_1=min(j_1,l_t_1)

            if j_1-in_1-1>=0:
                tmp[:,0:in_1]=x_l[:,(j_1-1):(j_1-1-in_1):-1]
            else:
                tmp[:,0:in_1]=x_l[:,(j_1-1)::-1]

            for m in range(n_area):
                z[m,j]=simps(tmp[m,:]*hrf,t_1)
        #e1=np.sum((y[:,(n1+1):(row_n-n1)]-z[:,(n1+1):(row_n-n1)])**2)
        e1=np.sum((y-z)**2)
        return e1,x_l[:,::int(1/fold)]
    def error_2(gamma,A,B,C,D):
        e2=0
        tmp_0=0
        for j in range(J):
            tmp_0=tmp_0+np.dot(np.dot(B[:,:,j],gamma),Q3[:,:,j])
        tmp=np.dot(gamma,Q1)-np.dot(np.dot(A,gamma),Q2)-tmp_0-np.dot(C,Q4)-np.repeat(D,l_t,axis=1) 
        for m in range(n_area):
            e2=e2+simps(tmp[m,:]**2,t_i)
        return e2
    def penalty(gamma,A,B,C,D):
        plt=0
        for k in range(n_area):
            w_1k=np.dot(np.dot(gamma[k,:],P5),gamma[k,:])**0.5
            plt=plt+np.sum(abs(A[:,k]))*w_1k
            for j in range(J):
                w_2kj=np.dot(np.dot(gamma[k,:],P10[:,:,j,j]),gamma[k,:])**0.5
                plt=plt+np.sum(abs(B[:,k,j]))*w_2kj
        for k in range(J):
            w_3k=(P14[k,k])**0.5
            plt=plt+np.sum(abs(C[:,k]))*w_3k
        
        return plt
    E1=list()
    E2=list()
    DENS=list()
    PLT=list()
    X=list()
    for i in range(len(results)):
        A=results[i][2]
        B=results[i][3]
        C=results[i][4]
        D=results[i][5]
        gamma=results[i][1]
        e1,x=error_1(A,B,C,D,gamma)
        e2=error_2(gamma,A,B,C,D)
        plt=penalty(gamma,A,B,C,D)
        dens=1.0*np.sum(abs(A)>0)/(A.shape[0]*A.shape[1])
        E1.append(e1)
        E2.append(e2)
        PLT.append(plt)
        X.append(x)
        DENS.append(dens)
        print results[i][0],e1,e2,plt,dens
        
    ind=np.argsort(E1)[0]
    print E1[ind]
    r_ind=ind
    #DENS_1=list()
    #E1_1=list()
    
    #r_ind=-1
    #for i in range(len(ind)):
    #    if r_ind!=-1:
    #        break
    #    DENS_1.append(DENS[ind[i]])
    #    E1_1.append(E1[ind[i]])
    #    if DENS[ind[i]]<1.2*np.sum(abs(config.A_real)>0)/(config.A_real.shape[0]*config.A_real.shape[1]):
    #        r_ind=ind[i]
    #        print 'haha'
    #        break
    #   if i>len(ind)/6.0 or DENS[ind[i]]<1.0/(n_area*n_area):
    #       r_ind=ind[np.argsort(DENS_1)[0]] 
    return r_ind,X[r_ind]
