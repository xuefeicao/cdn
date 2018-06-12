"""
An example with two subjects fMRI signals with same stimuli, increase number of subjects will improve the results
In our paper, we do 50 subjects. Here, to illustrate how to use our package, we only include two subjects.
Data are included in the folder data/
"""
import sys
#sys.path.append('../..')
#from cdn_analysis import cdn_multi_sub
#from evaluation_1 import eva
from cdn.cdn_analysis import cdn_multi_sub
from cdn.evaluation_1 import eva 

folder_name = ['Analysis/1/', 'Analysis/2/']
data_file = ['data/fMRI_1.txt', 'data/fMRI_2.txt']
real_parameter_file = 'data/real.pkl' # dict for real para
stimuli_folder = ['data/stimuli/']*2 # This folder only contains evt files
dt = 0.72 #TR of fMRI 
cdn_multi_sub(folder_name, data_file, stimuli_folder, val_pair=(0,1), dt=dt, lam=[0.1,0.01,1,10,50,100], mu=[0], lam_1=[0], tol=1e-2, max_iter=100, N=50, fold=0.5, share_stimuli=True, num_cores=2)
eva(folder_name, real_parameters=real_parameter_file, saved_folder_name='')
