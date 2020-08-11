from SIOBSDA_lung import SIOBSDA_main
import numpy as np
from warnings import warn
import pandas as pd
from scipy.io import loadmat
import os

os.environ['CUDA_VISIBLE_DEVICES'] = "1"

def run_SIOBSDA(lr_N=0.001,lr_V=0.000001,size_NN=[200,250,200],tilde_M=50,opt_iters=1000):
    '''
    #Sample run file for a lung-lung experiment setup.
    #The main code gets the following inputs:
    #N_train: A list of numpy arrays containing the counts for domains (target domain last), rows corresponding to genes and columns corresponding to samples.
    #X_train: A list of numpy arrays containing the labels for domains (target domain last).
    #N_test: A numpy array containing the counts for test data, rows corresponding to genes and columns corresponding to samples.
    #X_test: A numpy array containing the labels for test data. (For error estimation)
    #output_file: The name for the file to save the results..
    #lr_N: The base learning rate for mixing distribution parameters (NN parameters, omega).
    #lr_V: The base learning rate for variational parameters (varepsilon).
    #size_NN: The list containing the hidden layer sizes for the NN used in the mixing distribution.
    #tilde_M: Number of psi samples for ELBO calculation.
    #opt_iters: Number of optimization iterations.
    #indices_phi: A tuple of arrays corresponding to nodes indices connected in the interactome.
    #class_prior_ratio_1: Class probabilities for target domain.
    '''

    #Reading data for a specific lung-lung example
    adj_g = np.load('/DATA/genes_genes_net.npy')
    ind = np.squeeze(pd.read_csv('/DATA/Lung_netids_prior.csv',header=None).values)
    adj_g = adj_g[ind][:,ind,:]

    n_t_0 = 162
    n_t_1 = 240
    n_s_0 = 576
    n_s_1 = 552
    percent_train = 0.05
    percent_source = 0.1
    att0 = pd.read_csv('/DATA/Lung_allexpression.csv').values
    att0=att0[:,2:]
    att0=att0.astype(np.float32)
    ind_de = pd.read_csv('/DATA/Lung_logfc_1.csv').values
    ind_de = ind_de.astype(np.float32)
    ind_de_sorted=np.argsort(ind_de[:,1])[::-1]
    att0 = att0[ind_de_sorted[0:2500:5],:]

    adj_g = adj_g[ind_de_sorted[0:2500:5]][:,ind_de_sorted[0:2500:5],:]
    adj_label=(1.0*(adj_g[:,:,0]+adj_g[:,:,1]+adj_g[:,:,2]+adj_g[:,:,3]+adj_g[:,:,4]+adj_g[:,:,5]).astype(np.bool)).astype(np.int64)
    adj_label_tril=np.tril(adj_label,-1)
    indices_phi = np.nonzero(adj_label_tril)

    #read index here
    indices = loadmat('/DATA/indices_lung_1.mat') 
    ind_t1 = np.squeeze(indices['ind_t1'])-1
    ind_t2 = np.squeeze(indices['ind_t2'])-1
    ind_s1 = np.squeeze(indices['ind_s1'])-1
    ind_s2 = np.squeeze(indices['ind_s2'])-1
    luad_1 = att0[:,0:n_t_0]
    luad_1_train = luad_1[:,ind_t1[0:int(np.floor(percent_train*n_t_0))]]
    luad_1_test = luad_1[:,ind_t1[int(np.floor(percent_train*n_t_0)):]]

    lusc_1 = att0[:,n_t_0:n_t_0+n_t_1]
    lusc_1_train = lusc_1[:,ind_t2[0:int(np.floor(percent_train*n_t_1))]]
    lusc_1_test = lusc_1[:,ind_t2[int(np.floor(percent_train*n_t_1)):]]

    luad_2_all = att0[:,n_t_0+n_t_1:n_t_0+n_t_1+n_s_0]
    luad_2 = luad_2_all[:,ind_s1[0:int(np.floor(percent_source*n_s_0))]]

    lusc_2_all = att0[:,n_t_0+n_t_1+n_s_0:]
    lusc_2 = lusc_2_all[:,ind_s2[0:int(np.floor(percent_source*n_s_1))]]

    N_train = []
    N_train.append(np.concatenate((luad_2,lusc_2),axis=1))#Append Source
    N_train.append(np.concatenate((luad_1_train,lusc_1_train),axis=1))#Append Target
    N_test = np.concatenate((luad_1_test,lusc_1_test),axis=1)#Target test data
    X_train = []
    X_train.append(np.concatenate((np.ones(np.shape(luad_2)[1]),2*np.ones(np.shape(lusc_2)[1]))))#Append Source
    X_train.append(np.concatenate((np.ones(np.shape(luad_1_train)[1]),2*np.ones(np.shape(lusc_1_train)[1]))))#Append Target

    X_test = np.concatenate((np.ones(np.shape(luad_1_test)[1]),2*np.ones(np.shape(lusc_1_test)[1])))#Target test labels

    class_prior_ratio_0 = np.array([n_s_0/float(n_s_0+n_s_1),n_s_1/float(n_s_0+n_s_1)])#Source
    class_prior_ratio_1 = np.array([n_t_0/float(n_t_0+n_t_1),n_t_1/float(n_t_0+n_t_1)])#Target

    
    output_file = 'save_results_lung_lung_1'   
    
    SIOBSDA_main(N_train,X_train,N_test,X_test,class_prior_ratio_1,indices_phi,lr_N,lr_V,size_NN,tilde_M,output_file,opt_iters)


run_SIOBSDA()

