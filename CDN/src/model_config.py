import rpy2.robjects as robjects
import numpy as np
import math
class Modelpara(object):
    def __init__(self,pre_result):

        robjects.r.load(pre_result)
        self.P1=np.array(robjects.r['x'][0])
        self.P2=np.array(robjects.r['x'][1])
        self.P3=np.array(robjects.r['x'][2])
        self.P4=np.array(robjects.r['x'][3])
        self.P5=np.array(robjects.r['x'][4])
        self.P6=np.array(robjects.r['x'][5])
        self.P7=np.array(robjects.r['x'][6])
        self.P8=np.array(robjects.r['x'][7])
        self.P9=np.array(robjects.r['x'][8])
        self.P10=np.array(robjects.r['x'][9])
        self.P11=np.array(robjects.r['x'][10])
        self.P12=np.array(robjects.r['x'][11])
        self.P13=np.array(robjects.r['x'][12])
        self.P14=np.array(robjects.r['x'][13])
        self.P15=np.array(robjects.r['x'][14])
        self.Q1=np.array(robjects.r['x'][15])
        self.Q2=np.array(robjects.r['x'][16])
        self.Q3=np.array(robjects.r['x'][17])
        self.Q4=np.array(robjects.r['x'][18])
    
        self.Omega=np.array(robjects.r['x'][19])
        self.t_1=np.array(robjects.r['x'][20])
        self.hrf=np.array(robjects.r['x'][21])
        self.t_U=np.array(robjects.r['x'][22])
        self.t=np.array(robjects.r['x'][23])

        self.Q1_all=np.array(robjects.r['x'][24])
        self.Q2_all=np.array(robjects.r['x'][25])
        self.Q3_all=np.array(robjects.r['x'][26])
        self.Q4_all=np.array(robjects.r['x'][27])
        self.t_all=np.array(robjects.r['x'][28])
        self.t_U_all=np.array(robjects.r['x'][29])

        self.row_n=int(np.array(robjects.r['x'][30])[0])
        self.J=int(np.array(robjects.r['x'][31])[0])
        self.N=int(np.array(robjects.r['x'][32])[0])
        self.p=int(np.array(robjects.r['x'][33])[0])
        self.dt=np.array(robjects.r['x'][34])[0]
        self.fold=np.array(robjects.r['x'][35])[0]

        self.l_t_1=self.t_1.shape[0]
      
        
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
    def __init__(self,pre_result,sim_index):
        robjects.r.load(pre_result)
        self.y=np.array(robjects.r['x'][0])

        self.n_area=self.y.shape[0]
        self.multi=False
        self.sim_index=sim_index
        self.A_u=True
        self.B_u=True
        self.C_u=True
        self.D_u=False
        
