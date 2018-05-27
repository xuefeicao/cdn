import numpy as np
import seaborn as sns
from six.moves import cPickle as pickle
import matplotlib
matplotlib.pyplot.switch_backend('agg')
import matplotlib.pyplot as plt
import glob
import os
file_name_y_all=glob.glob('*.rdata')
for i in range(len(file_name_y_all)):
    file_name_y=file_name_y_all[i]
    print file_name_y
    f_tmp= 'results/'+file_name_y.replace('.rdata','')+'/config/'
    f=open(f_tmp+'summary_1.pickle','rb')
    save=pickle.load(f)
    summary=save['summary']
    [A_real,B_real,C_real,D_real]=summary['real A, B, C, D']
    x=summary['x']
    estimated_x=summary['estimated_x']
    estimated_y=summary['estimated_y']
    y=summary['y']
    E=summary['E']
    x_AB=summary['x_AB']
    [A,B,C,D]=summary['estimated A, B, C, D']
    f_tmp=f_tmp+'plot_summary/'
    if not os.path.exists(f_tmp):
        os.makedirs(f_tmp)
    print summary['lamu']
    row_n=y.shape[1]
    E_real=summary['E_real']
    t=summary['t']
    n1=summary['n1']
    #print n1,estimated_x.shape,estimated_y.shape,x.shape,y.shape
    t=t[(n1+1):(row_n-n1)]
    estimated_x=estimated_x[:,(n1+1):(row_n-n1)]
    estimated_y=estimated_y[:,(n1+1):(row_n-n1),:]
    y=y[:,(n1+1):(row_n-n1),:]
    x=x[:,(n1+1):(row_n-n1),:]
    x_AB=x_AB[:,(n1+1):(row_n-n1)]

    #plot A,A_real
    a_r = sns.heatmap(A_real,xticklabels=False,yticklabels=False)
    a_r.set_xlabel('A_real')
    fig=a_r.get_figure()
    fig.savefig(f_tmp+'A_real.png')

    plt.figure()
    #a_e = sns.heatmap(1.0*np.sum(abs(A)>0,axis=2)/A.shape[2],xticklabels=False,yticklabels=False)
    a_e = sns.heatmap(A[:,:,0],xticklabels=False,yticklabels=False)
    a_e.set_xlabel('A_estimate')
    fig=a_e.get_figure()
    fig.savefig(f_tmp+'A_estimate.png')

    plt.figure()
    a_r = sns.heatmap(B_real[:,:,0],xticklabels=False,yticklabels=False)
    a_r.set_xlabel('B_real')
    fig=a_r.get_figure()
    fig.savefig(f_tmp+'B_real.png')

    plt.figure()
    a_e = sns.heatmap(B[:,:,0,0],xticklabels=False,yticklabels=False)
    a_e.set_xlabel('B_estimate')
    fig=a_e.get_figure()
    fig.savefig(f_tmp+'B_estimate.png')

    plt.figure()
    a_r = sns.heatmap(C_real[:,:],xticklabels=False,yticklabels=False)
    a_r.set_xlabel('C_real')
    fig=a_r.get_figure()
    fig.savefig(f_tmp+'C_real.png')

    plt.figure()
    a_e = sns.heatmap(C[:,:,0],xticklabels=False,yticklabels=False)
    a_e.set_xlabel('C_estimate')
    fig=a_e.get_figure()
    fig.savefig(f_tmp+'C_estimate.png')



    #plot x, estimated x , four area, subplot 

    area=[0,1,2,3]
    f, axarr = plt.subplots(2, 2)
    axarr[0, 0].plot(t,x[area[0],:,0],color='r',label='real')
    axarr[0,0].plot(t,estimated_x[area[0],:,0],color='b',label='estimated')
    #axarr[0,0].plot(t,x_AB[area[0],:],color='b',label='estimated')
    axarr[0, 0].set_title(str(area[0])+'th channel neural activity')
    axarr[0,0].legend(loc='upper left')

    axarr[0, 1].plot(t,x[area[1],:,0],color='r')
    axarr[0,1].plot(t,estimated_x[area[1],:,0],color='b')
    #axarr[0,1].plot(t,x_AB[area[1],:],color='b',label='estimated')
    axarr[0, 1].set_title(str(area[1])+'th channel neural activity')

    axarr[1, 0].plot(t,x[area[2],:,0],color='r')
    axarr[1,0].plot(t,estimated_x[area[2],:,0],color='b')
    #axarr[1,0].plot(t,x_AB[area[2],:],color='b',label='estimated')
    axarr[1, 0].set_title(str(area[2])+'th channel neural activity')

    axarr[1, 1].plot(t,x[area[3],:,0],color='r')
    axarr[1,1].plot(t,estimated_x[area[3],:,0],color='b')
    #axarr[1,1].plot(t,x_AB[area[3],:],color='b',label='estimated')
    axarr[1, 1].set_title(str(area[3])+'th channel neural activity')
    f.savefig(f_tmp+'Neural.png')


    #plot y, estimated_y
    area=[0,1,2,3]
    f, axarr = plt.subplots(2, 2)
    axarr[0, 0].plot(t,y[area[0],:,0],color='r',label='real')
    axarr[0,0].plot(t,estimated_y[area[0],:,0],color='b',label='estimated')
    axarr[0, 0].set_title(str(area[0])+'th channel BOLD signal')
    axarr[0,0].legend(loc='upper left')

    axarr[0, 1].plot(t,y[area[1],:,0],color='r')
    axarr[0,1].plot(t,estimated_y[area[1],:,0],color='b')
    axarr[0, 1].set_title(str(area[1])+'th channel BOLD signal')

    axarr[1, 0].plot(t,y[area[2],:,0],color='r')
    axarr[1,0].plot(t,estimated_y[area[2],:,0],color='b')
    axarr[1, 0].set_title(str(area[2])+'th channel BOLD signal')

    axarr[1, 1].plot(t,y[area[3],:,0],color='r')
    axarr[1,1].plot(t,estimated_y[area[3],:,0],color='b')
    axarr[1, 1].set_title(str(area[3])+'th channel BOLD signal')

    f.savefig(f_tmp+'BOLD.png')


    #plot error compare, box chart #######################

    f=plt.figure()
    plt.hist(E[:,0])
    plt.ylabel('E1')
    plt.plot([E_real[0,0]], [0], marker='o', markersize=6, color="red")
    f.savefig(f_tmp+'E1.png')


    f=plt.figure()
    plt.hist(E[:,1])
    plt.ylabel('E2')
    plt.plot([E_real[0,1]], [0], marker='o', markersize=6, color="red")
    f.savefig(f_tmp+'E2.png')


    f=plt.figure()
    plt.hist(E[:,2])
    plt.ylabel('penalty')
    plt.plot([E_real[0,2]], [0], marker='o', markersize=3, color="red")
    f.savefig(f_tmp+'plt.png')
    ########################################################
    #print(E[:,0])
    #print(E_real[0,0])
    #print(E[:,1])
    #print(E_real[0,1])
    #print(E[:,2])
    #print(E_real[0,2])
    #plot bias st chart, A, B, C, D
    #plot f1,f2, report 
    #to do set up different threshold
    print summary['met']

