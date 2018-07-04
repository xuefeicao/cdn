from cdn.model_config import Modelconfig, Modelpara
from cdn.main_computation import update_p, select_lamu 
from cdn.data_preprocess import data_prepare
from functools import partial
import numpy as np
import multiprocessing as mp
import os
from sys import getsizeof
import six 
from six.moves import cPickle as pkl
import glob
import matplotlib.pyplot
matplotlib.pyplot.switch_backend('agg')
import matplotlib.pyplot as plt


def cdn_multi_sub(folder_name, data_file, stimuli_folder, val_pair, dt, lam, mu=[0], lam_1=[0], N=50, fold=0.5, share_stimuli=True, max_iter=100, tol=1e-2, num_cores=1, B_u=True, C_u=True, plot_r=True):

    """
    CDN analysis main function

    Parameters
    -----------
    folder_name: list of strings, used to save all data and analysis results
                 after running this function, you will get a structure like the following: 
                 for each folder_name
                                       / data: save all data files 
                                       / para: save results for tuning 
                                       / results: save all results with pictures 
    data_file: list of files name, BOLD signal for corresponding subject
    stimuli_folder: list of stimuli foders for all subject
                    if the subjects share the same stimuli, let every entry of stimuli_folder be the same
    val_pair: tuple of indices which are used to do cross validation, for example, (0,1) means we will select 
              data_file[0] to compute the different estimations for all tuning parameters and based on data_file[1] to select
              tuning parameters.  
    dt: TR of fMRI data
    lam, mu, lam_1: tuning parameters, lam*mu is the l2 penalty for the parameters. lam_1 is the penalty for the second derivative of
                    x (neuronal activities) which is commonly used in Functional Data Analysis
    N: num of basis - 1, 
    fold: scalar (integral evaluation stepsize = fold*dt)
    x_real, A_real, B_real, C_real: the real parameters which are used to verify simulations 

    share_stimuli: bool variable, if the subjects have the same stimuli, set this as True
    #share_tuning: bool variable, if you want only want to do tuning parameter selection for one subject and other subjects
                  use this selected parameter, set this as True. 
    tol, max_iter: parameter for algorithm convergence
    num_cores: int, number of cores for parallel computing 
    B_u, C_u: whether to update B and C in the estimations, bool variables
    plot_r: indicates whether to plot estimated signals, bool variables

    Returns
    ----------
    None, results saved in the folder
    """
    n1, n2, n3 = len(folder_name), len(data_file), len(stimuli_folder)
    if len(set([n1,n2,n3])) != 1:
        raise Exception('Please check the input files, you need the same length for folder_name, data_file and stimuli_folder')
    if n1 < 2:
        raise Exception('less than two simulations/subjects')
    precomp_dir = [0]*n1 
    val_data_dir = None 
    val_precomp_dir = None 
    ind1, ind2 = val_pair
    
    
    for i in range(n1):
        if folder_name[i][-1] != '/':
            folder_name = folder_name[i] + '/'
        pickle_file_config = folder_name[i] + 'results/'
        pickle_file_data= folder_name[i] + 'data/'
        pickle_file_para= folder_name[i] +'para/'
        if not os.path.exists(pickle_file_data):
            os.makedirs(pickle_file_data)
        if not os.path.exists(pickle_file_config):
            os.makedirs(pickle_file_config)
        if not os.path.exists(pickle_file_para):
            os.makedirs(pickle_file_para)
        if share_stimuli and i == ind1:
            data_prepare(data_file[ind1], stimuli_folder[ind1], pickle_file_data, dt, N, fold, precomp=True)
            precomp_dir[i] = folder_name[ind1] + 'data/'
            val_precomp_dir = pickle_file_data
        elif share_stimuli:
            data_prepare(data_file[i], stimuli_folder[i], pickle_file_data, dt, N, fold, precomp=False)
            precomp_dir[i] = folder_name[ind1] + 'data/'
        else:
            data_prepare(data_file[i], stimuli_folder[i], pickle_file_data, dt, N, fold, precomp=True)
            precomp_dir[i] = folder_name[i] + 'data/'

    if not val_precomp_dir:
        val_precomp_dir = folder_name[ind2] + 'data/'
    val_data_dir = folder_name[ind2] + 'data/'
    
    config = select_lamu(lam, mu, lam_1, folder_name[ind1], folder_name[ind1] +'para/', precomp_dir[ind1], val_data_dir, val_precomp_dir, num_cores=num_cores, tol=tol, max_iter=max_iter)   
    lam, mu, lam_1 = config.lamu

    # to be done: parallel 

    for i in range(n1):
        if not os.path.exists(folder_name[i]+'results/result.pkl'):
            update_p(folder_name[i], precomp_dir=precomp_dir[i], pickle_file=None, tol=tol, max_iter=max_iter, multi=False, lamu=[lam,mu,lam_1])

    
    # plot the results
    if plot_r:
        for i in range(n1):
            with open(folder_name[i]+'results/result.pkl', 'rb') as f:
                if six.PY2:
                    save = pkl.load(f)
                else:
                    save = pkl.load(f, encoding='latin1')
            t = save['t']
            n1 = save['n1'] 
            estimated_x = save['estimated_x']
            estimated_y = save['estimated_y']
            y = save['y']
            row_n = y.shape[1]
            # cut off begining and end
            t = t[(n1+1):(row_n-n1)]
            for j in range(config.y.shape[0]):
                f, axarr = plt.subplots(2, 1)
                axarr[0].plot(t, estimated_x[j,(n1+1):(row_n-n1)], color='red', label='estimated', linewidth=2)
                #axarr[0].set_xlabel('Time (Sec)')
                axarr[1].set_title('Neuronal activities')
                axarr[1].plot(t, estimated_y[j,(n1+1):(row_n-n1)], color='red', label='estimated', linewidth=2)
                axarr[1].plot(t, y[j,(n1+1):(row_n-n1)], color='blue', label='real', linewidth=2)
                axarr[1].set_xlabel('Time (Sec)')
                axarr[1].set_title('BOLD signal')
                plt.subplots_adjust(hspace=0.25)
                f.savefig(folder_name[i] + 'results/estimated_' + str(j)+ '.svg')
                plt.close()       
