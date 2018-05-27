import numpy as np
from sklearn import metrics
from sklearn.metrics import roc_auc_score
import math
import bootstrapped.bootstrap as bs
import bootstrapped.stats_functions as bs_stats
from sklearn.covariance import GraphLasso
import nitime
import nitime.analysis as nta
import nitime.timeseries as ts
import nitime.utils as tsu
def fd_2(A1,A):
    if np.sum(abs(A1))<1e-6:
        return [-1,-1,-1,-1]
    if len(A1.shape)==3 and np.sum(abs(A1)>0)==(A1.shape[0]*A1.shape[1]*A1.shape[2]):
        return [-1,-1,-1,-1]
    if len(A1.shape)==2 and np.sum(abs(A1)>0)==(A1.shape[0]*A1.shape[1]):
        return [-1,-1,-1,-1]
    n=A.shape[-1]
    m1=A.shape[0]
    m2=A.shape[1]
    A1=abs(A1)
    A=abs(A)
    if len(A.shape)==3:
        tmp=np.zeros((m1,m2))
        for i in range(m1):
            for j in range(m2):
                tmp[i,j]=sum(A[i,j,:]>0)
    else:
        tmp=np.zeros((m1,m2,A.shape[2]))
        for i in range(m1):
            for j in range(m2):
                for k in range(A.shape[2]):
                    tmp[i,j,k]=sum(A[i,j,k,:]>0)
    tmp=1.0*tmp/n 
    A1=(A1>0)
    fpr,tpr,thresholds=metrics.roc_curve(A1.reshape((-1)),tmp.reshape((-1)))
    sr=roc_auc_score(A1.reshape((-1)),tmp.reshape((-1)))
    return (fpr,tpr, thresholds, sr, tmp)
def fd_3(A1,A):
    if np.sum(abs(A1))<1e-6:
        return [-1,-1,-1,-1]
    if len(A1.shape)==3 and np.sum(abs(A1)>0)==(A1.shape[0]*A1.shape[1]*A1.shape[2]):
        return [-1,-1,-1,-1]
    if len(A1.shape)==2 and np.sum(abs(A1)>0)==(A1.shape[0]*A1.shape[1]):
        return [-1,-1,-1,-1]
    n_area=A1.shape[0]
    A1=abs(A1)
    if len(A.shape)==3:
        A=abs(np.mean(A,axis=2))
        tmp=np.zeros(((n_area*(n_area-1))/2,1))
        tmp_1=np.zeros(((n_area*(n_area-1))/2,1))
        for i in range(n_area):
            for j in range(i):
                tmp[(i*(i-1))/2+j,0]=max(A[i,j],A[j,i])
                tmp_1[(i*(i-1))/2+j,0]=max(A1[i,j],A1[j,i])
    else:
        A=abs(np.mean(A,axis=3))
        J=A.shape[2]
        tmp=np.zeros(((n_area*(n_area-1))/2,J))
        tmp_1=np.zeros(((n_area*(n_area-1))/2,J))
        for i in range(n_area):
            for j in range(i):
                for k in range(J):
                    tmp[(i*(i-1))/2+j,k]=max(A[i,j,k],A[j,i,k])
                    tmp_1[(i*(i-1))/2+j,k]=max(A1[i,j,k],A1[j,i,k])
    tmp=tmp/max(tmp)
    tmp_1=(tmp_1>0)

    fpr,tpr,thresholds=metrics.roc_curve(tmp_1.reshape((-1)),tmp.reshape((-1)))
    sr=roc_auc_score(tmp_1.reshape((-1)),tmp.reshape((-1)))
    return (fpr,tpr, thresholds, sr, tmp)
def eva_smr(config,configpara,rc_1=True):
    smr={}
    n_area=config[0].n_area
    row_n=configpara.row_n
    J=configpara.J
    fold=configpara.fold 
    p=configpara.p
    P12=configpara.P12
    Q2=configpara.Q2_all
    #Q2=configpara.Q2
    Q2=Q2[:,0:(Q2.shape[1]+1):int(1/fold)]
    A_real=config[0].A_real
    B_real=config[0].B_real
    C_real=config[0].C_real
    D_real=config[0].D_real

    
    A=np.zeros((n_area,n_area,len(config)))
    B=np.zeros((n_area,n_area,J,len(config)))
    C=np.zeros((n_area,J,len(config)))
    D=np.zeros((n_area,1,len(config)))
    estimated_x=np.zeros((n_area,row_n,len(config)))
    estimated_y=np.zeros((n_area,row_n,len(config)))
    estimated_gamma=np.zeros((n_area,p,len(config)))
    E=np.zeros((len(config),3))

    y=np.zeros((n_area,row_n,len(config)))
    x=np.zeros((n_area,row_n,len(config)))
    gamma_real=np.zeros((n_area,p,len(config)))
    E_real=np.zeros((len(config),3))

    for i in range(len(config)):
        A[:,:,i]=config[i].A
        B[:,:,:,i]=config[i].B
        C[:,:,i]=config[i].C
        D[:,:,i]=config[i].D
        estimated_gamma[:,:,i]=config[i].gamma 
        estimated_y[:,:,i]=np.dot(config[i].gamma,np.transpose(P12))
        estimated_x[:,:,i]=np.dot(config[i].gamma,Q2)
        x[:,:,i]=config[i].x
        y[:,:,i]=config[i].y
        gamma_real[:,:,i]=config[i].gamma_real
        E[i,0]=config[i].e1
        E[i,1]=config[i].e2
        E[i,2]=config[i].plt 
        E_real[i,0]=config[i].e1_r
        E_real[i,1]=config[i].e2_r
        E_real[i,2]=config[i].plt_r
    if rc_1==True:
        l=[fd_2(A_real,A),fd_2(B_real,B),fd_2(C_real,C)]
        l1=[fd_3(A_real,A),fd_3(B_real,B)]

    smr['E_real']=E_real
    smr['E']=E 
    smr['lamu']=config[0].lamu 
    smr['estimated A, B, C, D']=[A,B,C,D]
    smr['real A, B, C, D']=[A_real,B_real,C_real,D_real]
    smr['x']=x
    smr['y']=y 
    smr['estimated_x']=estimated_x
    smr['estimated_y']=estimated_y
    smr['estimated_gamma']=estimated_gamma
    smr['real_gamma']=gamma_real
    smr['x_AB']=config[0].x_AB
    smr['t']=np.arange(0,configpara.dt*(configpara.row_n-1)+configpara.dt**0.5,configpara.dt)
    smr['n1']=(math.floor(configpara.t_i[0]/configpara.dt)+1)
    smr['roc_1']=l
    smr['roc_2']=l1    
    

    return smr 



        

