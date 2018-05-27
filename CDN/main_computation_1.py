import numpy as np
import multiprocessing as mp
import time
from scipy.integrate import simps
from functools import partial
from validation_truncation_2_1 import cross_validation
from model_config import Modelconfig, Modelpara
import os
from six.moves import cPickle as pickle
import random
import glob
def update_p(sim_n,file_name_dir,pickle_file,pickle_file_sim,lamu):
    configpara=Modelpara(file_name_dir+'precomp_'+str(sim_n)+'.rdata')
    config=Modelconfig(file_name_dir+'observed_'+str(sim_n)+'.rdata',sim_n)
    if sim_n==0:
        config.multi=True
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
    Omega=configpara.Omega
    y=config.y 
    n_area=config.n_area
    p=configpara.p
    t_i=configpara.t_i
    l_t=configpara.l_t
    J=configpara.J 
    t_T=configpara.t_T 
    dt=configpara.dt
    row_n=configpara.row_n
    fold=configpara.fold
    ###################################################################################
    def gr(gamma,A,B,C,D,lam,mu,lam_1):
        g=np.zeros((n_area,p))
        g=g+np.dot(gamma,P1)-np.dot(np.dot(np.transpose(A),gamma),np.transpose(P2))
        g=g-np.dot(np.dot(A,gamma),P2)+np.dot(np.dot(np.dot(np.transpose(A),A),gamma),P5)
        tmp_1=0
        tmp_2=0
        for j in range(J):
            tmp_1=tmp_1+np.dot(np.dot(B[:,:,j],gamma),P3[:,:,j])
            tmp_2=tmp_2+np.dot(np.dot(np.dot(np.transpose(A),B[:,:,j]),gamma),P6[:,:,j])
        g=g-(tmp_1-tmp_2)
        g=g-np.dot(C,P4)+np.dot(np.dot(np.transpose(A),C),P7)
        g=g-np.dot(D,P8)+np.dot(np.dot(np.transpose(A),D),P9)
        tmp=0
        for l in range(J):
            tmp_1=0
            for j in range(J):
                tmp_1=np.dot(np.dot(B[:,:,j],gamma),P10[:,:,j,l])
            tmp=tmp-np.dot(np.transpose(B[:,:,l]),(np.dot(gamma,np.transpose(P3[:,:,l]))-np.dot(np.dot(A,gamma),np.transpose(P6[:,:,l]))-tmp_1-np.dot(C,P13[:,:,l])-np.dot(D,P11[l,:].reshape((1,-1)))))
        g=g+tmp
        g=g*2*lam
        tmp1=np.zeros((n_area,1))
        tmp2=np.zeros((n_area,J))

        for m in range(n_area):
            tmp1[m,0]=np.sum(abs(A[:,m]))/np.dot(np.dot(gamma[m,:],P5),gamma[m,])**0.5
            for j in range(J):
                tmp2[m,j]=np.sum(abs(B[:,m,j]))/np.dot(np.dot(gamma[m,:],P10[:,:,j,j]),gamma[m,:])**0.5
        g=g+lam*mu*np.dot(gamma,np.transpose(P5))*tmp1
        for j in range(J):
            g=g+lam*mu*np.dot(gamma,P10[:,:,j,j])*(tmp2[:,j].reshape((-1,1)))
        g=g+np.dot((np.dot(gamma,np.transpose(P12))-y),P12)*2
        g=g+2*lam_1*np.dot(gamma,np.transpose(Omega))
        g[np.isnan(g)]=0
        return g 
    def cd_thre(tmp,tmp_1,mu):
        #print(abs(tmp),mu*(tmp_1**0.5))
        mu=mu/2.0
        if abs(tmp)>mu*(tmp_1**0.5):
            return np.sign(tmp)*(abs(tmp)-mu*(tmp_1**0.5))/tmp_1
        else:
            return 0
    def update_A(m,n,gamma,A,B,C,D,mu):
        tmp_0=0
        for j in range(J):
            tmp_0=tmp_0+np.dot(np.dot(np.dot(B[m,:,j],gamma),P6[:,:,j]),gamma[n,:])
        tmp_1=np.dot(np.dot(gamma[n,:],P5),gamma[n,:])
        tmp=np.dot(np.dot(gamma[n,:],P2),gamma[m,:])-np.dot(np.dot(np.dot(A[m,:],gamma),P5),gamma[n,:])-tmp_0-np.dot(np.dot(C[m,:],P7),gamma[n,:])-D[m,0]*np.dot(gamma[n,:],P9[0,:])+A[m,n]*tmp_1
        return cd_thre(tmp,tmp_1,mu)
    def update_B(m,n,j,gamma,A,B,C,D,mu):
        tmp_0=0
        for l in range(J):
            tmp_0=tmp_0+np.dot(np.dot(np.dot(B[m,:,l],gamma),P10[:,:,l,j]),gamma[n,:])
        tmp_1=np.dot(np.dot(gamma[n,:],P10[:,:,j,j]),gamma[n,:])
        tmp=np.dot(np.dot(gamma[n,:],P3[:,:,j]),gamma[m,:])-np.dot(np.dot(np.dot(A[m,:],gamma),np.transpose(P6[:,:,j])),gamma[n,:])-tmp_0-np.dot(np.dot(C[m,:],P13[:,:,j]),gamma[n,:])-D[m,0]*np.dot(gamma[n,:],P11[j,:])+B[m,n,j]*tmp_1
        return cd_thre(tmp,tmp_1,mu)
    def update_C(m,n,gamma,A,B,C,D,mu):
        tmp_0=0
        for j in range(J):
            tmp_0=tmp_0+np.dot(np.dot(B[m,:,j],gamma),P13[n,:,j])
        tmp_1=P14[n,n]
        tmp=np.dot(gamma[m,:],P4[n,:])-np.dot(np.dot(A[m,:],gamma),P7[n,:])-tmp_0-np.dot(C[m,:],P14[n,:])-D[m,0]*P15[0,n]+C[m,n]*tmp_1
        return cd_thre(tmp,tmp_1,mu)
    def update_D(gamma,A,B,C):
        tmp=np.dot(gamma,np.transpose(P8))-np.dot(np.dot(A,gamma),np.transpose(P9))
        for j in range(J):
            tmp=tmp-np.dot(np.dot(B[:,:,j],gamma),P11[j,:]).reshape((-1,1))
        tmp=tmp-np.dot(C,np.transpose(P15))
        return tmp*1.0/t_T
    def likelihood(gamma,A,B,C,D,lam,mu,lam_1,p_t=False):
        e1=np.sum((y-np.dot(gamma,np.transpose(P12)))**2)
        e2=0
        tmp_0=0
        for j in range(J):
            tmp_0=tmp_0+np.dot(np.dot(B[:,:,j],gamma),Q3[:,:,j])
        tmp=np.dot(gamma,Q1)-np.dot(np.dot(A,gamma),Q2)-tmp_0-np.dot(C,Q4)-np.repeat(D,l_t,axis=1) 
        for m in range(n_area):
            e2=e2+simps(tmp[m,:]**2,t_i)
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
        plt_1=0
        for i in range(n_area):
            plt_1=plt_1+np.dot(np.dot(gamma[i,:],Omega),gamma[i,:])
        
        sum_e=e1+lam*e2+lam*mu*plt+lam_1*plt_1
        if p_t==True:
            #print(e1,e2,plt)
            return(e1,e2,plt,plt_1)
        return sum_e
    def update_all_3(gamma,mu):
        n_all=n_area*(J+1)+J+1
        Y_tmp=np.zeros((n_area,n_all))
        X_tmp=np.zeros((n_all,n_all))
        I_tmp=np.zeros((n_all,n_all))
        W_A=np.zeros((n_area,n_area))
        for i in range(n_area):
             W_A[i,i]=np.dot(np.dot(gamma[i,:],P5),np.transpose(gamma[i,:]))
        I_tmp[0:n_area,0:n_area]=W_A
        W_B=np.zeros((n_area,n_area,J))
        for j in range(J):
            for i in range(n_area):
                W_B[i,i,j]=np.dot(np.dot(gamma[i,:],P10[:,:,j,j]),np.transpose(gamma[i,:]))
            I_tmp[((j+1)*n_area):((j+2)*n_area),((j+1)*n_area):((j+2)*n_area)]=W_B[:,:,j]
        W_C=np.zeros((J,J))
        for j in range(J):
            W_C[j,j]=P14[j,j]
        I_tmp[((J+1)*n_area):((J+1)*n_area+J),((J+1)*n_area):((J+1)*n_area+J)]=W_C
        for j in range(J+1):
            if j==0:
                Y_tmp[:,j*n_area:((j+1)*n_area)]=np.dot(np.dot(gamma,np.transpose(P2)),np.transpose(gamma))
            else:
                Y_tmp[:,j*n_area:((j+1)*n_area)]=np.dot(np.dot(gamma,np.transpose(P3[:,:,j-1])),np.transpose(gamma))
        Y_tmp[:,((J+1)*n_area):((J+1)*n_area+J)]= np.dot(gamma,np.transpose(P4))
        Y_tmp[:,-1]=np.dot(gamma,np.transpose(P8)).reshape((-1,))
        for j in range(J+1):
            if j==0:
                X_tmp[j*n_area:((j+1)*n_area),0:n_area]=np.dot(np.dot(gamma,P5),np.transpose(gamma))
            else:
                X_tmp[j*n_area:((j+1)*n_area),0:n_area]=np.dot(np.dot(gamma,P6[:,:,j-1]),np.transpose(gamma))
        X_tmp[((J+1)*n_area):((J+1)*n_area+J),0:n_area]= np.dot(P7,np.transpose(gamma))
        X_tmp[-1,0:n_area]=np.dot(P9,np.transpose(gamma))
        tmp=np.zeros((n_area*J,n_area*J))
        for j in range(J):
            for l in range(J):
                tmp[j*n_area:(j+1)*n_area,l*n_area:(l+1)*n_area]=np.dot(np.dot(gamma,P10[:,:,j,l]),np.transpose(gamma))
        for j in range(J):
            X_tmp[0:n_area,((j+1)*n_area):((j+2)*n_area)]=np.dot(np.dot(gamma,np.transpose(P6[:,:,j])),np.transpose(gamma))
            X_tmp[(J+1)*n_area:(J+1)*n_area+J,((j+1)*n_area):((j+2)*n_area)]=np.dot(P13[:,:,j],np.transpose(gamma))
            X_tmp[-1,((j+1)*n_area):((j+2)*n_area)]=np.dot(P11[j,:].reshape((1,-1)),np.transpose(gamma))
        X_tmp[n_area:(J+1)*n_area,n_area:(J+1)*n_area]=tmp
        ##C
        X_tmp[0:n_area,(J+1)*n_area:((J+1)*n_area+J)]=np.dot(gamma,np.transpose(P7))
        for j in range(J):
            X_tmp[n_area*(j+1):n_area*(j+2),(J+1)*n_area:((J+1)*n_area+J)]=np.dot(gamma,np.transpose(P13[:,:,j]))
        X_tmp[(J+1)*n_area:((J+1)*n_area+J),(J+1)*n_area:((J+1)*n_area+J)]=P14
        X_tmp[-1,(J+1)*n_area:((J+1)*n_area+J)]=P15
        ##D
        X_tmp[0:n_area,-1]=np.dot(gamma,np.transpose(P9)).reshape((-1))
        for j in range(J):
            X_tmp[n_area*(j+1):n_area*(j+2),-1]=np.dot(gamma,np.transpose(P11[j,:])).reshape((-1))
        X_tmp[(J+1)*n_area:((J+1)*n_area+J),-1]=np.transpose(P15).reshape((-1))
        X_tmp[-1,-1]=t_T 
        s_eig=np.sort(abs(np.linalg.eig(X_tmp)[0]))
        #print np.linalg.cond(X_tmp+mu*I_tmp), s_eig[-1] ,s_eig[0]
        if config.D_u==False:
            Y_tmp=Y_tmp[:,0:-1]
            X_tmp=X_tmp[0:-1,0:-1]
            I_tmp=I_tmp[0:-1,0:-1]
            return np.dot(Y_tmp,np.linalg.pinv(X_tmp+mu*I_tmp))
        return np.dot(Y_tmp,np.linalg.pinv(X_tmp+mu*I_tmp))

    def update_all_2(gamma,mu):
        n_all=n_area+J+1
        Y_tmp=np.zeros((n_area,n_all))
        X_tmp=np.zeros((n_all,n_all))
        I_tmp=np.zeros((n_all,n_all))
        W_A=np.zeros((n_area,n_area))
        for i in range(n_area):
             W_A[i,i]=np.dot(np.dot(gamma[i,:],P5),np.transpose(gamma[i,:]))
        I_tmp[0:n_area,0:n_area]=W_A
        W_C=np.zeros((J,J))
        for j in range(J):
            W_C[j,j]=P14[j,j]
        I_tmp[((1)*n_area):((1)*n_area+J),((1)*n_area):((1)*n_area+J)]=W_C
        Y_tmp[:,0:n_area]=np.dot(np.dot(gamma,np.transpose(P2)),np.transpose(gamma))
        Y_tmp[:,((1)*n_area):((1)*n_area+J)]= np.dot(gamma,np.transpose(P4))
        Y_tmp[:,-1]=np.dot(gamma,np.transpose(P8)).reshape((-1,))
        X_tmp[0:n_area,0:n_area]=np.dot(np.dot(gamma,P5),np.transpose(gamma))
        X_tmp[((1)*n_area):((1)*n_area+J),0:n_area]= np.dot(P7,np.transpose(gamma))
        X_tmp[-1,0:n_area]=np.dot(P9,np.transpose(gamma))
        ##C
        X_tmp[0:n_area,n_area:(n_area+J)]=np.dot(gamma,np.transpose(P7))
        X_tmp[n_area:(n_area+J),n_area:(n_area+J)]=P14
        X_tmp[-1, n_area: (n_area+J)]=P15
        ##D
        X_tmp[0:n_area,-1]=np.dot(gamma,np.transpose(P9)).reshape((-1))
        X_tmp[n_area:(n_area+J),-1]=np.transpose(P15).reshape((-1))
        X_tmp[-1,-1]=t_T 
        s_eig=np.sort(abs(np.linalg.eig(X_tmp)[0]))
        if config.D_u==False:
            Y_tmp=Y_tmp[:,0:-1]
            X_tmp=X_tmp[0:-1,0:-1]
            I_tmp=I_tmp[0:-1,0:-1]
            return np.dot(Y_tmp,np.linalg.pinv(X_tmp+mu*I_tmp))
        return np.dot(Y_tmp,np.linalg.pinv(X_tmp+mu*I_tmp))

    def update_all_1(gamma,mu):
        n_all=n_area+1
        Y_tmp=np.zeros((n_area,n_all))
        X_tmp=np.zeros((n_all,n_all))
        I_tmp=np.zeros((n_all,n_all))
        W_A=np.zeros((n_area,n_area))
        for i in range(n_area):
             W_A[i,i]=np.dot(np.dot(gamma[i,:],P5),np.transpose(gamma[i,:]))
        I_tmp[0:n_area,0:n_area]=W_A
  
        Y_tmp[:,0:n_area]=np.dot(np.dot(gamma,np.transpose(P2)),np.transpose(gamma))
  
        Y_tmp[:,-1]=np.dot(gamma,np.transpose(P8)).reshape((-1,))


        X_tmp[0:n_area,0:n_area]=np.dot(np.dot(gamma,P5),np.transpose(gamma))
      
        X_tmp[-1,0:n_area]=np.dot(P9,np.transpose(gamma))
        ##D
        X_tmp[0:n_area,-1]=np.dot(gamma,np.transpose(P9)).reshape((-1))
        X_tmp[-1,-1]=t_T 
        s_eig=np.sort(abs(np.linalg.eig(X_tmp)[0]))
        #print np.linalg.cond(X_tmp+mu*I_tmp), s_eig[-1] ,s_eig[0]
        if config.D_u==False:
            Y_tmp=Y_tmp[:,0:-1]
            X_tmp=X_tmp[0:-1,0:-1]
            I_tmp=I_tmp[0:-1,0:-1]
            return np.dot(Y_tmp,np.linalg.pinv(X_tmp+mu*I_tmp))
        return np.dot(Y_tmp,np.linalg.pinv(X_tmp+mu*I_tmp))


    #######################################################################################
    ##############################################################################################
    def error_ws_0(gamma_ini,lam_1):
        e1=np.sum((y-np.dot(gamma_ini,np.transpose(P12)))**2)
        plt_1=0
        for i in range(n_area):
            plt_1=plt_1+np.dot(np.dot(gamma_ini[i,:],Omega),gamma_ini[i,:])
        return e1+lam_1*plt_1
    
    def error_ws(gamma_ini,lam_1):
        stp=1
        while(stp<5000):
            gr=np.dot((np.dot(gamma_ini,np.transpose(P12))-y),P12)*2+2*lam_1*np.dot(gamma_ini,np.transpose(Omega))
            n_gr=(np.sum(gr**2))
            f_t=1
            fixed=error_ws_0(gamma_ini,lam_1)
            while(error_ws_0(gamma_ini-f_t*gr,lam_1)>fixed-0.5*f_t*n_gr):
                f_t=0.8*f_t
            gamma_ini=gamma_ini-gr*f_t
            stp=stp+1
            if n_gr**0.5<0.001:
                break
        return gamma_ini

    def ini_select(lam_1):
        gamma_0=np.zeros((n_area,p))
        #gamma_0 = np.random.normal(size=(n_area,p))
        gamma_0=error_ws(gamma_0,lam_1)
        return gamma_0,0,0
        t_tmp=np.arange(0,dt*(row_n-1)+dt*fold*0.5,dt)
        Q2_tmp=Q2[:,0:(Q2.shape[1]+1):int(1/fold)]
        Q4_tmp=Q4[:,0:(Q4.shape[1]+1):int(1/fold)]
        x_tmp=np.dot(gamma_0,Q2_tmp)
        n_tmp=x_tmp.shape[1]-np.where(np.amax(abs(Q4_tmp),axis=0)>0)[0][-1]
        n_tmp_1=np.where(np.amax(abs(Q4_tmp),axis=0)>0)[0][0]
        m_tmp=int(n_tmp*p/row_n)
        m_tmp_1=int(n_tmp_1*p/row_n)
        for i in range(x_tmp.shape[0]):
            model=np.polyfit(t_tmp[-20:-10],x_tmp[i,-20:-10],1)
            p_tmp=np.polyval(model,t_tmp[-10:])
            x_tmp[i,-10:]=p_tmp
        gamma_0=np.transpose(np.dot(np.dot(np.linalg.inv(np.dot(Q2_tmp,np.transpose(Q2_tmp))),Q2_tmp),np.transpose(x_tmp)))
        gamma_0[:,-1]=np.sign(gamma_0[:,-1])*np.minimum(np.amax(abs(gamma_0[:,0:(p-1)]),axis=1),abs(gamma_0[:,-1]))
       
        return gamma_0,m_tmp,m_tmp_1
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
    lam=lamu[0]
    mu=lamu[1]/lamu[0]
    mu_1=mu
    mu_2=mu
    lam_1=lamu[4]
    A=np.zeros((n_area,n_area))
    B=np.zeros((n_area,n_area,J))
    C=np.zeros((n_area,J))
    D=np.zeros((n_area,1))
    iter=1
    sum_e=10**6
    gamma,n1,n2=ini_select(lam_1) 
    sum_e_1=likelihood(gamma,A,B,C,D,lam,mu,lam_1,p_t=True)[1]

    while(iter<100 and abs(sum_e-sum_e_1)/sum_e_1>0.01):
        gamma_1=gamma+np.ones((n_area,p))
        stp=1
        while(stp<10):
            results = gr(gamma,A,B,C,D,lam,mu,lam_1)
            n_results=(np.sum(results**2))
            gamma_1=gamma.copy()
            f_t=1
            fixed=likelihood(gamma,A,B,C,D,lam,mu,lam_1)
            while(likelihood(gamma-f_t*results,A,B,C,D,lam,mu,lam_1)>fixed-0.5*f_t*n_results):
                f_t=0.8*f_t
            gamma=gamma-results*f_t
            stp=stp+1
            #if (n_results**0.5<0.001):
             #   break
        if config.B_u==True:
            tmp=update_all_3(gamma,mu=0)
            A=tmp[:,0:n_area]
            for j in range(J):
                B[:,:,j]=tmp[:,((j+1)*n_area):((j+2)*n_area)]
            C=tmp[:,((J+1)*n_area):((J+1)*n_area+J)]
            if config.D_u==True:
                D=tmp[:,-1].reshape((-1,1))
        elif config.C_u==True:
            tmp=update_all_2(gamma,mu=0)
            A=tmp[:,0:n_area]
            C=tmp[:,n_area:(n_area+J)]
            if config.D_u==True:
                D=tmp[:,-1].reshape((-1,1))
        else:
            tmp=update_all_1(gamma,mu=0)
            A=tmp[:,0:n_area]
            if config.D_u==True:
                D=tmp[:,-1].reshape((-1,1))
        sum_e=sum_e_1
        sum_e_1=likelihood(gamma,A,B,C,D,lam,mu,lam_1,p_t=True)[1]
        iter=iter+1
    e1,e2,plt,plt_1=likelihood(gamma,A,B,C,D,lam,mu,lam_1,p_t=True)
    print config.sim_index,lamu,config.sim_index,iter, 'dif_sum='+str(abs(sum_e-sum_e_1)), e1, e2, plt, plt_1 

    if config.multi==False:
        config.gamma=gamma
        config.A=A
        config.B=B
        config.C=C
        config.D=D
        config.lamu=lamu
        config.e1=e1
        config.e2=e2
        config.plt=plt 
        config.plt_1=plt_1
        config.t_i=configpara.t_i
        os.system('echo 1') 
        pickle_file_1=pickle_file_sim+str(config.sim_index)+'.pickle'
        f = open(pickle_file_1, 'wb')
        save = {
        'config': config
        }
        pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
        f.close()
        return
    else:
        
        pickle_file_1=pickle_file+str_1(lam)+'_'+str_1(mu)+'_'+str_1(mu_1)+'_'+str_1(mu_2)+'_'+str(lam_1)+'.pickle'
        os.system('echo 0')
        f = open(pickle_file_1, 'wb')
        save = {
        'result': [lamu,gamma,A,B,C,D,e1,e2,plt,plt_1]
        }
        pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
        f.close()
        return 
def str_2(num):
    if num[0]=='0':
	return float(num)/(10**(len(num)-1))
    else:
	return float(num)
    
def select_lamu(sim_n,sim_n_1,lam,mu,mu_1,mu_2,lam_1,file_name_dir,pickle_file,pickle_file_sim):
    para=list()
    file_config=os.listdir(pickle_file)
    lamu_done=np.zeros((len(file_config),5))
    for i in range(len(file_config)):
        num=str.split(file_config[i].replace('.pickle',''),'_')
        num_0=str_2(num[0])
        num_1=str_2(num[1])
        num_2=str_2(num[2])
        num_3=str_2(num[3])
        num_4=str_2(num[4])
        lamu_done[i,:]=[num_0,num_1,num_2,num_3,num_4]
    print lamu_done
    for i in range(len(lam)):
        for j in range(len(mu)):
            for k in range(len(lam_1)):
                for l in range(len(mu_1)):
                    for o in range(len(mu_2)):
                        if lamu_done.shape[0]==0 or min(np.sum(abs(lamu_done-[lam[i],mu[j],mu[j],mu[j],lam_1[k]]),axis=1))>1e-9:
                            para.append((lam[i],lam[i]*mu[j],lam[i]*mu_1[l],lam[i]*mu_2[o],lam_1[k]))
    if len(para)>=1:
        pool = mp.Pool(processes=16)
        update_p_1=partial(update_p,sim_n,file_name_dir,pickle_file,pickle_file_sim)
        pool.map(update_p_1,para)
        pool.close()
        pool.join()
    results=list()
    file_config=glob.glob(pickle_file+'*.pickle')
    for i in range(len(file_config)):
        f = open(file_config[i], 'rb')
        save=pickle.load(f)
        results.append(save['result'])

    config_1_para=Modelpara(file_name_dir+'precomp_'+str(sim_n_1)+'.rdata')
    config_1=Modelconfig(file_name_dir+'observed_'+str(sim_n_1)+'.rdata',sim_n_1)
    ind,x=cross_validation(config_1,config_1_para,results)
    config=Modelconfig(file_name_dir+'observed_'+str(sim_n)+'.rdata',sim_n)
    config.t_i=config_1_para.t_i
    config.x_AB=x
    config.lamu=results[ind][0]
    config.A=results[ind][2]
    config.B=results[ind][3]
    config.C=results[ind][4]
    config.D=results[ind][5]
    config.gamma=results[ind][1]
    config.e1=results[ind][6]
    config.e2=results[ind][7]
    config.plt=results[ind][8]
    config.plt_1=results[ind][9]


    return config





   

