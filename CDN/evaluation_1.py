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
def bst(A):
    tmp_0=np.zeros(A.shape[0:-1])
    if len(A.shape)==3:
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                tmp=bs.bootstrap(A[i,j,:], stat_func=bs_stats.mean,num_iterations=20000,alpha=0.05)
                #print A1[i,j]
                #print tmp
                if 0<tmp.lower_bound or 0>tmp.upper_bound:
                    tmp_0[i,j]=1

    else:
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                for k in range(A.shape[2]):
                    tmp=bs.bootstrap(A[i,j,k,:], stat_func=bs_stats.mean,num_iterations=20000,alpha=0.1)
                    if 0<tmp.lower_bound or 0>tmp.upper_bound:
                        tmp_0[i,j,k]=1
    return tmp_0

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

   
    A=np.zeros((n_area,n_area,len(config)))
    B=np.zeros((n_area,n_area,J,len(config)))
    C=np.zeros((n_area,J,len(config)))
    D=np.zeros((n_area,1,len(config)))
    estimated_x=np.zeros((n_area,row_n,len(config)))
    estimated_y=np.zeros((n_area,row_n,len(config)))
    estimated_gamma=np.zeros((n_area,p,len(config)))
    E=np.zeros((len(config),3))
    y=np.zeros((n_area,row_n,len(config)))
    for i in range(len(config)):
        A[:,:,i]=config[i].A
        B[:,:,:,i]=config[i].B
        C[:,:,i]=config[i].C
        D[:,:,i]=config[i].D
        estimated_gamma[:,:,i]=config[i].gamma 
        estimated_y[:,:,i]=np.dot(config[i].gamma,np.transpose(P12))
        estimated_x[:,:,i]=np.dot(config[i].gamma,Q2)
        y[:,:,i]=config[i].y
        E[i,0]=config[i].e1
        E[i,1]=config[i].e2
        E[i,2]=config[i].plt 



    smr['E']=E 
    smr['lamu']=config[0].lamu 
    smr['estimated A, B, C, D']=[A,B,C,D]
    smr['y']=y 
    smr['estimated_x']=estimated_x
    smr['estimated_y']=estimated_y
    smr['estimated_gamma']=estimated_gamma
    smr['x_AB']=config[0].x_AB
    smr['t']=np.arange(0,configpara.dt*(configpara.row_n-1)+configpara.dt*0.5,configpara.dt)
    smr['n1']=(int(configpara.t_i[0]/configpara.dt)+1)
    smr['bst_A']=bst(A)
    smr['bst_B']=bst(B)
    smr['bst_C']=bst(C)

    

    return smr 



        

