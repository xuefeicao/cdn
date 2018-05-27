#add two level class instead one level to save memorys
import sys
sys.path.append('../code_utility')
from model_config import Modelconfig,Modelpara
from main_computation_1 import update_p, select_lamu 
from functools import partial
from evaluation import eva_smr
import numpy as np
import multiprocessing as mp
import os
from sys import getsizeof
from six.moves import cPickle as pickle
import glob
sim_n=50

file_name_y_all=glob.glob('*.rdata')
lam=[0.1,0.01,1,10,50,100]
#mu=[0.1,0.01,0.5,1,5,10,30,50,100]
mu=[0]
mu_1=[1]
mu_2=[1]
lam_1=[0]
for i in range(len(file_name_y_all)):
    file_name_y=file_name_y_all[i]
    file_name='../precomp_'+file_name_y.split('_')[-1].replace('.rdata','')+'_c.rdata'
    configpara=Modelpara(file_name)
    pickle_file_config ='results/'+file_name_y.replace('.rdata','')+'/config/'
    pickle_file_sim='results/'+file_name_y.replace('.rdata','')+'/sim/'
    pickle_file_para='results/'+file_name_y.replace('.rdata','')+'/para/'
    if not os.path.exists(pickle_file_sim):
        os.makedirs(pickle_file_sim)
    if not os.path.exists(pickle_file_config):
        os.makedirs(pickle_file_config)
    if not os.path.exists(pickle_file_para):
        os.makedirs(pickle_file_para)
    #pickle_file=pickle_file_config+'config_lamu_1.pickle'
    pickle_file=pickle_file_sim+'0.pickle'
    if os.path.exists(pickle_file):
        f = open(pickle_file, 'rb')
        save=pickle.load(f)
        config=save['config']
    else:
        config=Modelconfig(file_name_y,0)
        config.multi=True
        config=select_lamu(0,configpara,1,lam,mu,mu_1,mu_2,lam_1,file_name_y,pickle_file_para,pickle_file_sim)
        f = open(pickle_file, 'wb')
        save = {
           'config': config,
            }
        pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
        f.close()
    lamu=config.lamu
    config_all=list()

    pickle_file=pickle_file_config+'config_all_1.pickle'
    if os.path.exists(pickle_file):
        f = open(pickle_file, 'rb')
        save=pickle.load(f)
        config_all=save['config_all']
    else:
        file_config=os.listdir(pickle_file_sim)
        trials_done=np.zeros((len(file_config),1))
        for i in range(len(file_config)):
            num=int(file_config[i].replace('.pickle',''))
            trials_done[i,:]=num
        para=list()
        for i in range(1,sim_n):
            if i not in trials_done:
                para.append(i)
        if len(para)>=1:
            pool = mp.Pool(processes=16)
            update_p_1=partial(update_p,configpara=configpara,file_name_y=file_name_y,pickle_file=pickle_file_para,pickle_file_sim=pickle_file_sim,lamu=lamu)
            print('all computation')
            pool.map(update_p_1,para)
        #config_all.append(config)
        for i in range(sim_n):
            f = open(pickle_file_sim+str(i)+'.pickle', 'rb')
            save=pickle.load(f)
            config_all.append(save['config'])
        f = open(pickle_file, 'wb')
        save = {
        'config_all': config_all,
        }
        pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
        f.close()

    summary_all=eva_smr(config_all,configpara)
    pickle_file =pickle_file_config+ 'summary_1.pickle'
    f = open(pickle_file, 'wb')
    save = {
       'summary': summary_all,
        }
    pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
    f.close()




    
    


