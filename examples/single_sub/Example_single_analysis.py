import sys
sys.path.append('../../CDN/')
from CDN_analysis import CDN_fmri

folder_name = 'Analysis/'
data_file = 'data/fMRI.txt'
stimuli_folder = 'data/stimuli/' # This folder only contains evt files
dt = 0.72 #TR of fMRI 
CDN_fmri(folder_name, data_file, stimuli_folder, dt, lam=[0.1,0.01,1,10,50,100], mu=[0], lam_1=[0], tol=1e-2, max_iter=100, N=50, fold=0.5, data_dir=None, x_real=None, A_real=None, B_real=None, C_real=None)
