from __future__ import print_function
import numpy as np
import math
from six.moves import cPickle as pkl 
import six 
class Modelpara(object):
    """
    creat precomputed data
    """
    def __init__(self,pre_result):
        with open(pre_result, 'rb') as f:
            if six.PY2:
                save = pkl.load(f)
            else:
                save = pkl.load(f, encoding='latin1')
        # for k in save:
        #     if hasattr(save[k], 'shape'):
        #         print(k, save[k].shape)
        self.P1 = save['P1']
        self.P2 = save['P2']
        self.P3 = save['P3']
        self.P4 = save['P4']
        self.P5 = save['P5']
        self.P6 = save['P6']
        self.P7 = save['P7']
        self.P8 = save['P8']
        self.P9 = save['P9']
        self.P10 = save['P10']
        self.P11 = save['P11']
        self.P12 = save['P12']
        self.P13 = save['P13']
        self.P14 = save['P14']
        self.P15 = save['P15']
        self.Q1 = save['Q1']
        self.Q2 = save['Q2']
        self.Q3 = save['Q3']
        self.Q4 = save['Q4']
    
        self.Omega = save['Omega']
        self.t_1 = save['t_1']
        self.hrf = save['hrf']
        self.t_U = save['t_U']
        self.t = save['t']

        self.Q1_all = save['Q1_all']
        self.Q2_all = save['Q2_all']
        self.Q3_all = save['Q3_all']
        self.Q4_all = save['Q4_all']
        self.t_all = save['t_all']
        self.t_U_all = save['t_U_1']

        self.row_n = int(save['row_n'])
        self.J = int(save['J'])
        self.N = int(save['N'])
        self.p = int(save['p'])
        self.dt = save['dt']
        self.fold = save['fold']

        self.l_t_1 = self.t_1.shape[0]
      
        
        #r_n=math.floor(2*self.dt*(self.row_n-1)/(self.N*self.dt*self.fold))
        #self.t_i=np.arange(0,self.dt*(self.row_n-1)+self.dt*self.fold*0.5,self.dt*self.fold)
        #self.l_t=self.t_i.shape[0]
        self.t_i_all=self.t_all
        self.t_i=self.t
        self.l_t_all=self.t_i_all.shape[0]
        self.l_t=self.t_i.shape[0]
 
        self.t_T=self.t[-1]-self.t[0]
        self.t_T_all=self.t_all[-1]-self.t_all[0]


class Modelconfig(object):
    """
    creat model configuration
    """
    def __init__(self, pre_result, A_u=True, B_u=True, C_u=True):
        """
        Parameters
        ------------
        pre_result: name of observed data
        A_u, B_u, C_u: bool variable which indicates whether you want to update A, B, C 
        """
        with open(pre_result, 'rb') as f:
            if six.PY2:
                save = pkl.load(f)
            else:
                save = pkl.load(f, encoding='latin1')
        self.y = save['y']
        #self.A_real = save['A_real']
        #self.B_real = save['B_real']
        #self.C_real = save['C_real']
        #self.x_real = save['x_real']
        self.n_area = self.y.shape[0]
        self.A_u = A_u
        self.B_u = B_u
        self.C_u = C_u 
        self.D_u = False
        
