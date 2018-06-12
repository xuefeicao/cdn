from __future__ import print_function
import numpy as np
from sklearn import metrics
from sklearn.metrics import roc_auc_score
import math
import six 
#import bootstrapped.bootstrap as bs
#import bootstrapped.stats_functions as bs_stats
from six.moves import cPickle as pkl
#from sklearn.covariance import GraphLasso
#import nitime
#import nitime.analysis as nta
#import nitime.timeseries as ts
#import nitime.utils as tsu
# def bst_0(A, num_iterations=10000, alpha=0.05):
#     """
#     bootstrap estimation

#     Parameters
#     ----------
#     num_iterations: int, iterations for bootstrap
#     alpha: significant level

#     Returns
#     ---------
#     numpy array, bootstrapped estimations

#     """
#     tmp_0 = np.zeros(A.shape[0:-1])
#     if len(A.shape) == 3:
#         for i in range(A.shape[0]):
#             for j in range(A.shape[1]):
#                 tmp = bs.bootstrap(A[i,j,:], stat_func=bs_stats.mean,num_iterations=20000,alpha=0.05)
#                 if 0<tmp.lower_bound or 0>tmp.upper_bound:
#                     tmp_0[i,j] = 1

#     else:
#         for i in range(A.shape[0]):
#             for j in range(A.shape[1]):
#                 for k in range(A.shape[2]):
#                     tmp = bs.bootstrap(A[i,j,k,:], stat_func=bs_stats.mean,num_iterations=20000,alpha=0.1)
#                     if 0<tmp.lower_bound or 0>tmp.upper_bound:
#                         tmp_0[i,j,k] = 1
#     return tmp_0
def spl_mean(org, num_iterations):
    """
    bootstrap sampling

    Parameters
    ------------
    nums_iterations: int

    Returns
    ------------
    samples: list of means
    """
    l = [0]*num_iterations
    for i in range(num_iterations):
        l[i] = np.mean(np.random.choice(org, len(org)))
    return l 



def bst(A, num_iterations=10000, alpha=0.05):
    """
    bootstrap estimation of p values without using packages

    Parameters
    ----------
    num_iterations: int, iterations for bootstrap
    

    Returns
    ---------
    numpy array, bootstrapped p values
    numpy array, bootstrapped estimations

    """
    tmp_0 = np.zeros(A.shape[0:-1])
    para_mean = np.mean(A, axis=-1)
    if len(A.shape) == 3:
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                if para_mean[i,j] >= 0:
                    tmp_0[i,j] = np.mean(spl_mean(A[i,j,:]<=0, num_iterations))
                else:
                    tmp_0[i,j] = np.mean(spl_mean(A[i,j,:]>0, num_iterations))
                

    else:
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                for k in range(A.shape[2]):
                    if para_mean[i,j,k] >= 0:
                        tmp_0[i,j,k] = np.mean(spl_mean(A[i,j,k,:]<=0, num_iterations))
                    else:
                        tmp_0[i,j,k] = np.mean(spl_mean(A[i,j,k,:]>0, num_iterations))
    return tmp_0, (tmp_0 < alpha)*para_mean


def fd_0(A1,A):
    """
    compute AUC of estimated A if real A is known, used for simulated data

    Parameters
    ----------
    A1: numpy array, real A
    A: estimated data, all estimations from all subjects

    Returns
    ----------
    scalar, AUC
    """
    if np.sum(abs(A1)) < 1e-6:
        return -1
    if len(A1.shape) == 3 and np.sum(abs(A1)>0) == (A1.shape[0]*A1.shape[1]*A1.shape[2]):
        return -1
    if len(A1.shape) == 2 and np.sum(abs(A1)>0) == (A1.shape[0]*A1.shape[1]):
        return -1
    if len(A.shape) == 3:
        tmp = abs(np.mean(A,axis=2))
        
    else:
        tmp = abs(np.mean(A,axis=3))
    if np.max(tmp) == 0:
        return -1
    tmp = tmp/np.max(tmp)
    A1 = (abs(A1)>0)
    #fpr,tpr,thresholds = metrics.roc_curve(A1.reshape((-1)),tmp.reshape((-1)))
    sr = roc_auc_score(A1.reshape((-1)),tmp.reshape((-1)))
    return sr


def eva(folder_name, saved_folder_name=None, real_parameters=None, num_iterations=10000, alpha=0.1):
    """
    evaluation of estimations

    Parameters
    -----------
    folder_name: folder names for all subjects analysis, the same meaning as that in function cdn_multi_sub
    saved_folder_name: folder used to save bootstrapped estimations
    nums_iterations, alpha: bootstrap para
    """
    n = len(folder_name)

    for i in range(n):
        with open(folder_name[i]+'results/result.pkl', 'rb') as f:
            if six.PY2:
                save = pkl.load(f)
            else:
                save = pkl.load(f, encoding='latin1')
        A = save['A']
        B = save['B']
        C = save['C']
        if i == 0:
            A_all = np.zeros((A.shape[0], A.shape[1], n))
            B_all = np.zeros((B.shape[0], B.shape[1], B.shape[2], n))
            C_all = np.zeros((C.shape[0], C.shape[1], n))
        A_all[:,:,i] = A
        B_all[:,:,:,i] = B
        C_all[:,:,i] = C
    if real_parameters:
        with open(real_parameters, 'rb') as f:
            if six.PY2:
                save = pkl.load(f)
            else:
                save = pkl.load(f, encoding='latin1')
        A_real = save['A_real']
        B_real = save['B_real']
        C_real = save['C_real']
        auc_a = fd_0(A_real, A_all)
        auc_b = fd_0(B_real, B_all)
        auc_c = fd_0(C_real, C_all)
        print('AUC(A):{0}, AUC(B):{1}, AUC(C):{2}'.format(auc_a, auc_b, auc_c))
    

    if saved_folder_name:
        save = {}
        save['bst_A']=bst(A_all)
        save['bst_B']=bst(B_all)
        save['bst_C']=bst(C_all)

        with open(saved_folder_name+'bst.pkl', 'wb') as f:
            pkl.dump(save, f, pkl.HIGHEST_PROTOCOL)











