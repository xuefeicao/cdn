import numpy as np
import math 
from six.moves import cPickle as pkl 
function (x, param = NULL, verbose = TRUE) 
{
    if (is.null(param)) {
        if (verbose == TRUE) {
            warning("Default parameters for HRF are used")
        }
        param <- list()
        param$a1 <- 6
        param$a2 <- 12
        param$b1 <- 0.9
        param$b2 <- 0.9
        param$c <- 0.35
    }
    d1 <- param$a1 * param$b1
    d2 <- param$a2 * param$b2
    (x/d1)^param$a1 * exp(-(x - d1)/param$b1) - param$c * (x/d2)^param$a2 * 
        exp(-(x - d2)/param$b2)
}
def CanonicalHRF(x):
	a1, a2, b1, b2 = 6, 12, 0.9, 0.9 
	d1 = a1*b1
	d2 = a2*b2
	c = 
def data_prepare(y_name, u_name, file_name, dt, N=50, fold=0.5, precomp=True, x_real=None, y_real=None, A_real=None, B_real=None, C_real=None, sim_data=None):
	"""
	preprocess the data for CDN analysis, the function is prepared for single subject processing 

	Parameters
	------------
	y_name: file name of fMRI BOLD signal with string format
	u_name: folder name of fMRI stimuli which includes only stimuli file indexed from *.ev1 to *.evJ where J is the number of stimuli
            We require the colum of the file is the time dimension where row is the space dimension
    file_name: list of two strings (the file name we use to save our observed data and precomputed data)
    dt: TR of fMRI signal
    N: number of basis
    fold: scalar (integral evaluation stepsize = fold*dt)
    precomp: bool (Whether to do precomputation for this subject). This variable is only useful when we do multi-subjects computation. 
    x_real: file name of neuronal signal
    y_real: file name of real fMRI BOLD signal
    A_real, B_real, C_real: numpy matrices (real parameters) 
    sim_data: file name of simulated data which is provided for verification of algorithm. If this is provided, other related parameters will be overrided except N.
	

	Returns
	------------
	None, preprocessed data will be saved into a file
	"""
	if not sim_data:
		with open(sim_data) as f:
			save = pkl.load(f)['simulation']
			y_name = save['y']
			u_name = save['u']
			fold = save['fold']
			x_real = save['x_real']
			y_real = save['y_real']
			A_real = save['A_real']
			B_real = save['B_real']
			C_real = save['C_real']
	else:
		# n_area * row_n
		y = np.loadtxt(y_name).T
		u = np.loadtxt(u_name).T 
	n_are, row_n = y.shape
	J = u.shape[0]
	h = dt*fold
    t_T = dt*(row_n-1)
    dt_1 = t_T/N
    
    t_0 = [i*dt for i in range(row_n)]
    l_t_0 = row_n

    # cut off begining and end time sequences

    r_n = math.floor(2*dt_1/(dt*fold))
    l_t = int((dt*(row_n-1)-2*r_n*dt*fold)/(dt*fold))
    t = [r_n*dt*fold + i*dt*fold for i in range(l_t)]
    hrf_l = int(30/(dt*fold))
    t_1 = [dt*fold*i for i in range(hrf_l)]

    
t_1<-seq(0,dt*12,by=dt*fold)
hrf<-canonicalHRF(t_1)
l_t_1<-length(t_1)
sim_n<-5
n_area<-6



	
