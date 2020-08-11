

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import numpy as np
import os


from scipy.io import loadmat,savemat
import tensorflow as tf
from tensorflow.contrib.distributions import fill_triangular

import json
from timeit import default_timer as timer

from utils import log_lognormal, sample_ln, sample_n_e, sample_logitn, log_logitn,log_logitn2,log_sum_exp,give_obc_error






def SIOBSDA_main(N_train,X_train,N_test,X_test,class_prior,indices_cnst,lr_N,lr_V,size_NN,K_val,file_name,opt_iters):
    
    cnst_num = len(indices_cnst[0])

    slim=tf.contrib.slim

    def sample_hyper(noise_dim,K,z_dim,reuse=False): 
        with tf.variable_scope("hyper_q") as scope:
            if reuse:
                scope.reuse_variables()
            e2 = tf.random_normal(shape=[K,noise_dim])
        
        
            h2 = slim.stack(e2,slim.fully_connected,size_NN)

            mu = tf.reshape(slim.fully_connected(h2,z_dim,activation_fn=None,scope='implicit_hyper_mu'),[-1,z_dim])
        return mu

    def obc_valid_test_error(N_test,N_valid,X_test,X_valid,class_prior,phitheta10_test_tmp,phitheta11_test_tmp,nb_term_test10_tmp,nb_term_test11_tmp,nb_term_valid10_tmp,nb_term_valid11_tmp):
            regul_=0.0
            n_test = np.shape(N_test)[1]
            n_valid = np.shape(N_valid)[1]
            num_post_samp = np.shape(phitheta10_test_tmp)[0]
            phitheta101_test_tmp = class_prior[0]*phitheta10_test_tmp + class_prior[1]*phitheta11_test_tmp


            logp10_test =  np.zeros((num_post_samp,n_test))
            logp11_test =  np.zeros((num_post_samp,n_test))
            logp1010_test =  np.zeros((num_post_samp,n_test))
            logp1011_test =  np.zeros((num_post_samp,n_test))
            logp10_valid =  np.zeros((num_post_samp,n_valid))
            logp11_valid =  np.zeros((num_post_samp,n_valid))
            logp1010_valid =  np.zeros((num_post_samp,n_valid))
            logp1011_valid =  np.zeros((num_post_samp,n_valid))

            for i in range(num_post_samp):    
                phitheta10_test = np.squeeze(phitheta10_test_tmp[i,:])
                phitheta11_test = np.squeeze(phitheta11_test_tmp[i,:])
                phitheta101_test = np.squeeze(phitheta101_test_tmp[i,:])
                p10_test=np.random.beta(g0+np.sum(N_test,axis=0),h0+np.sum(phitheta10_test))
                p11_test=np.random.beta(g0+np.sum(N_test,axis=0),h0+np.sum(phitheta11_test))
                p101_test=np.random.beta(g0+np.sum(N_test,axis=0),h0+np.sum(phitheta101_test))
                nb_term_test10_test = np.squeeze(nb_term_test10_tmp[:,:,i]) + N_test*np.expand_dims(np.log(p10_test+regul_),axis=0) + np.expand_dims(phitheta10_test,axis=1)*np.expand_dims(np.log(p10_test+regul_),axis=0)
                logp10_test[i,:] = np.sum(nb_term_test10_test,axis=0)
                nb_term_test11_test = np.squeeze(nb_term_test11_tmp[:,:,i]) + N_test*np.expand_dims(np.log(p11_test+regul_),axis=0) + np.expand_dims(phitheta11_test,axis=1)*np.expand_dims(np.log(p11_test+regul_),axis=0)
                logp11_test[i,:] = np.sum(nb_term_test11_test,axis=0)
                nb_term_test1010_test = np.squeeze(nb_term_test10_tmp[:,:,i]) + N_test*np.expand_dims(np.log(p101_test+regul_),axis=0) + np.expand_dims(phitheta10_test,axis=1)*np.expand_dims(np.log(p101_test+regul_),axis=0)
                logp1010_test[i,:] = np.sum(nb_term_test1010_test,axis=0)
                nb_term_test1011_test = np.squeeze(nb_term_test11_tmp[:,:,i]) + N_test*np.expand_dims(np.log(p101_test+regul_),axis=0) + np.expand_dims(phitheta11_test,axis=1)*np.expand_dims(np.log(p101_test+regul_),axis=0)
                logp1011_test[i,:] = np.sum(nb_term_test1011_test,axis=0)

                phitheta10_valid = np.squeeze(phitheta10_test_tmp[i,:])
                phitheta11_valid = np.squeeze(phitheta11_test_tmp[i,:])
                phitheta101_valid = np.squeeze(phitheta101_test_tmp[i,:])
                p10_valid=np.random.beta(g0+np.sum(N_valid,axis=0),h0+np.sum(phitheta10_valid))
                p11_valid=np.random.beta(g0+np.sum(N_valid,axis=0),h0+np.sum(phitheta11_valid))
                p101_valid=np.random.beta(g0+np.sum(N_valid,axis=0),h0+np.sum(phitheta101_valid))
                nb_term_test10_valid = np.squeeze(nb_term_valid10_tmp[:,:,i]) + N_valid*np.expand_dims(np.log(p10_valid+regul_),axis=0) + np.expand_dims(phitheta10_valid,axis=1)*np.expand_dims(np.log(p10_valid+regul_),axis=0)
                logp10_valid[i,:] = np.sum(nb_term_test10_valid,axis=0)
                nb_term_test11_valid = np.squeeze(nb_term_valid11_tmp[:,:,i]) + N_valid*np.expand_dims(np.log(p11_valid+regul_),axis=0) + np.expand_dims(phitheta11_valid,axis=1)*np.expand_dims(np.log(p11_valid+regul_),axis=0)
                logp11_valid[i,:] = np.sum(nb_term_test11_valid,axis=0)
                nb_term_test1010_valid = np.squeeze(nb_term_valid10_tmp[:,:,i]) + N_valid*np.expand_dims(np.log(p101_valid+regul_),axis=0) + np.expand_dims(phitheta10_valid,axis=1)*np.expand_dims(np.log(p101_valid+regul_),axis=0)
                logp1010_valid[i,:] = np.sum(nb_term_test1010_valid,axis=0)
                nb_term_test1011_valid = np.squeeze(nb_term_valid11_tmp[:,:,i]) + N_valid*np.expand_dims(np.log(p101_valid+regul_),axis=0) + np.expand_dims(phitheta11_valid,axis=1)*np.expand_dims(np.log(p101_valid+regul_),axis=0)
                logp1011_valid[i,:] = np.sum(nb_term_test1011_valid,axis=0)

        
            log_p_obc_test1 = []
            log_p_obc_test1.append(log_sum_exp(logp10_test))
            log_p_obc_test1.append(log_sum_exp(logp11_test))
            obc_error_test1 = np.array(give_obc_error(log_p_obc_test1,class_prior,X_test))
            log_p_obc_test2 = []
            log_p_obc_test2.append(log_sum_exp(logp1010_test))
            log_p_obc_test2.append(log_sum_exp(logp1011_test))
            obc_error_test2 = np.array(give_obc_error(log_p_obc_test2,class_prior,X_test))

            log_p_obc_valid1 = []
            log_p_obc_valid1.append(log_sum_exp(logp10_valid))
            log_p_obc_valid1.append(log_sum_exp(logp11_valid))
            obc_error_valid1 = np.array(give_obc_error(log_p_obc_valid1,class_prior,X_valid))
            log_p_obc_valid2 = []
            log_p_obc_valid2.append(log_sum_exp(logp1010_valid))
            log_p_obc_valid2.append(log_sum_exp(logp1011_valid))
            obc_error_valid2 = np.array(give_obc_error(log_p_obc_valid2,class_prior,X_valid))

            print("Valid OBC Error Setup 1:", obc_error_valid1[-1])
            print("Valid OBC Error Setup 2:", obc_error_valid2[-1])

            print("Test OBC Error Setup 1:", obc_error_test1[-1])
            print("Test OBC Error Setup 2:", obc_error_test2[-1])

            return obc_error_valid1[-1],obc_error_valid2[-1]

    def obc_test_error(N_test,N_valid,X_test,X_valid,class_prior,phitheta10_test_tmp,phitheta11_test_tmp,nb_term_test10_tmp,nb_term_test11_tmp):
            regul_=0.0
            n_test = np.shape(N_test)[1]
            n_valid = np.shape(N_valid)[1]
            num_post_samp = np.shape(phitheta10_test_tmp)[0]
            phitheta101_test_tmp = class_prior[0]*phitheta10_test_tmp + class_prior[1]*phitheta11_test_tmp


            logp10_test =  np.zeros((num_post_samp,n_test))
            logp11_test =  np.zeros((num_post_samp,n_test))
            logp1010_test =  np.zeros((num_post_samp,n_test))
            logp1011_test =  np.zeros((num_post_samp,n_test))


            for i in range(num_post_samp):    
                phitheta10_test = np.squeeze(phitheta10_test_tmp[i,:])
                phitheta11_test = np.squeeze(phitheta11_test_tmp[i,:])
                phitheta101_test = np.squeeze(phitheta101_test_tmp[i,:])
                p10_test=np.random.beta(g0+np.sum(N_test,axis=0),h0+np.sum(phitheta10_test))
                p11_test=np.random.beta(g0+np.sum(N_test,axis=0),h0+np.sum(phitheta11_test))
                p101_test=np.random.beta(g0+np.sum(N_test,axis=0),h0+np.sum(phitheta101_test))
                nb_term_test10_test = np.squeeze(nb_term_test10_tmp[:,:,i]) + N_test*np.expand_dims(np.log(p10_test+regul_),axis=0) + np.expand_dims(phitheta10_test,axis=1)*np.expand_dims(np.log(p10_test+regul_),axis=0)
                logp10_test[i,:] = np.sum(nb_term_test10_test,axis=0)
                nb_term_test11_test = np.squeeze(nb_term_test11_tmp[:,:,i]) + N_test*np.expand_dims(np.log(p11_test+regul_),axis=0) + np.expand_dims(phitheta11_test,axis=1)*np.expand_dims(np.log(p11_test+regul_),axis=0)
                logp11_test[i,:] = np.sum(nb_term_test11_test,axis=0)
                nb_term_test1010_test = np.squeeze(nb_term_test10_tmp[:,:,i]) + N_test*np.expand_dims(np.log(p101_test+regul_),axis=0) + np.expand_dims(phitheta10_test,axis=1)*np.expand_dims(np.log(p101_test+regul_),axis=0)
                logp1010_test[i,:] = np.sum(nb_term_test1010_test,axis=0)
                nb_term_test1011_test = np.squeeze(nb_term_test11_tmp[:,:,i]) + N_test*np.expand_dims(np.log(p101_test+regul_),axis=0) + np.expand_dims(phitheta11_test,axis=1)*np.expand_dims(np.log(p101_test+regul_),axis=0)
                logp1011_test[i,:] = np.sum(nb_term_test1011_test,axis=0)


        
            log_p_obc_test1 = []
            log_p_obc_test1.append(log_sum_exp(logp10_test))
            log_p_obc_test1.append(log_sum_exp(logp11_test))
            obc_error_test1 = np.array(give_obc_error(log_p_obc_test1,class_prior,X_test))
            log_p_obc_test2 = []
            log_p_obc_test2.append(log_sum_exp(logp1010_test))
            log_p_obc_test2.append(log_sum_exp(logp1011_test))
            obc_error_test2 = np.array(give_obc_error(log_p_obc_test2,class_prior,X_test))



            print("Test Best Vals OBC Error Setup 1:", obc_error_test1[-1])
            print("Test Best Vals OBC Error Setup 2:", obc_error_test2[-1])

            return obc_error_test1[-1],obc_error_test2[-1]



    regul_=0.0#1e-12




    
    n_test = np.shape(N_test)[1]
    dom_size=len(N_train)
    P=[]
    N=[]
    ind_train=[]
    N_lab=[]
    Lab=[]
    Lab_num=[]
    Lab_unique=[]
    n_train = []
    for i in range(dom_size):
        P_tmp,N_tmp = np.shape(N_train[i])    
        P.append(P_tmp)
        N.append(N_tmp)
        ind_train.append(X_train[i]!=0)
        n_train.append(np.sum(X_train[i]!=0))
        Lab.append(X_train[i][X_train[i]!=0])
        Lab_num.append(len(np.unique(X_train[i][X_train[i]!=0])))
        Lab_unique.append(np.unique(X_train[i][X_train[i]!=0]))
        N_lab.append(N_train[i][:,X_train[i]!=0])

    Labs_uniques=np.unique(np.concatenate(Lab_unique,axis=0).flatten())
    n_labs = len(Labs_uniques)
    N_training_1=[]
    n_size_1=[]
    N_training_2=[] 
    n_size_2=[]
    for i in range(dom_size):
        
            j=0
            N_training_1.append(N_train[i][:,X_train[i]==Lab_unique[i][j]])
            n_size_1.append(int(np.sum(X_train[i]==Lab_unique[i][j])))
            j=1
            N_training_2.append(N_train[i][:,X_train[i]==Lab_unique[i][j]])
            n_size_2.append(int(np.sum(X_train[i]==Lab_unique[i][j])))


    V = 32 
    
    



    g0=1e-2 
    h0=1e-2
    alpha0=.1
    beta0=.1
    w0=.01
    u0=.01
    e0=1.0
    f0=1.0
    a0=1.0
    d0=1.0
    aa0 = 0.05
    ab1 = 1

    tf.reset_default_graph()

    noise_dim = V*Lab_num[0] + V*Lab_num[1] + dom_size*V + n_labs + dom_size + V  + V*P[0]  + 2 
    psi_dim = V*Lab_num[0] + V*Lab_num[1] + dom_size*V + n_labs + dom_size + V  + V*P[0]  + 2 
    psi_dim_theta_0 = V*Lab_num[0]
    psi_dim_theta_1 = V*Lab_num[1]
    psi_dim_theta_each = V
    psi_dim_u = dom_size*V
    psi_dim_u_each = V
    psi_dim_nu = n_labs
    psi_dim_q = dom_size
    psi_dim_b = V
    psi_dim_phi = V*P[0]
    psi_dim_p_0 = n_train[0]
    psi_dim_p_1 = n_train[1]
    psi_dim_c = 1
    psi_dim_gamma = 1

    K = int(K_val)

    alpha_prior = tf.constant(0.05,shape=[P[0],V], name='prior_alpha')

    mu_phi_prior = 0.0*tf.log(alpha_prior)
    sigdirichlet_prior = ((P[0]-2.0)/(P[0]*alpha_prior) + (1.0)/(P[0]*alpha_prior))/10.0



    sigdirichlet_ = tf.get_variable("sig_dirichlet",shape=[1,P[0],V],dtype=tf.float32,initializer=tf.initializers.random_normal(mean=-6.9,stddev=0.1,seed=10))
    sigdirichlet = tf.exp(sigdirichlet_)

    sigtheta_00 = tf.get_variable("sig_theta00",shape=[1,V], dtype=tf.float32,initializer=tf.initializers.random_normal(mean=-5.0,stddev=0.1,seed=11))
    sigtheta00 = tf.exp(sigtheta_00)

    sigtheta_01 = tf.get_variable("sig_theta01",shape=[1,V], dtype=tf.float32,initializer=tf.initializers.random_normal(mean=-5.0,stddev=0.1,seed=12))
    sigtheta01 = tf.exp(sigtheta_01)

    sigtheta_10 = tf.get_variable("sig_theta10",shape=[1,V], dtype=tf.float32,initializer=tf.initializers.random_normal(mean=-5.0,stddev=0.1,seed=13))
    sigtheta10 = tf.exp(sigtheta_10)

    sigtheta_11 = tf.get_variable("sig_theta11",shape=[1,V], dtype=tf.float32,initializer=tf.initializers.random_normal(mean=-5.0,stddev=0.1,seed=14))
    sigtheta11 = tf.exp(sigtheta_11)

    sigu_0 = tf.get_variable("sig_u0",shape=[1,V], dtype=tf.float32,initializer=tf.initializers.random_normal(mean=-5.0,stddev=0.1,seed=15))
    sigu0 = tf.exp(sigu_0)

    sigu_1 = tf.get_variable("sig_u1",shape=[1,V], dtype=tf.float32,initializer=tf.initializers.random_normal(mean=-5.0,stddev=0.1,seed=16))
    sigu1 = tf.exp(sigu_1)

    signu_ = tf.get_variable("sig_nu",shape=[1,n_labs], dtype=tf.float32,initializer=tf.initializers.random_normal(mean=-5.0,stddev=0.1,seed=17))
    signu = tf.exp(signu_)

    sigb_ = tf.get_variable("sig_b",shape=[1,V], dtype=tf.float32,initializer=tf.initializers.random_normal(mean=-5.0,stddev=0.1,seed=18))
    sigb = tf.exp(sigb_)

    sigq_ = tf.get_variable("sig_q",shape=[1,dom_size], dtype=tf.float32,initializer=tf.initializers.random_normal(mean=-5.0,stddev=0.1,seed=19))
    sigq = tf.exp(sigq_)

    sigc_ = tf.get_variable("sig_c",shape=[1,1], dtype=tf.float32,initializer=tf.initializers.random_normal(mean=-5.0,stddev=0.1,seed=20))
    sigc = tf.exp(sigc_)

    siggamma_ = tf.get_variable("sig_gamma",shape=[1,1], dtype=tf.float32,initializer=tf.initializers.random_normal(mean=-5.0,stddev=0.1,seed=21))
    siggamma = tf.exp(siggamma_)

    sigp00_ = tf.get_variable("sig_p00",shape=[1,n_size_1[0]], dtype=tf.float32,initializer=tf.initializers.random_normal(mean=-5.0,stddev=0.1,seed=22))
    sigp00 = tf.exp(sigp00_)

    sigp01_ = tf.get_variable("sig_p01",shape=[1,n_size_2[0]], dtype=tf.float32,initializer=tf.initializers.random_normal(mean=-5.0,stddev=0.1,seed=23))
    sigp01 = tf.exp(sigp01_)

    sigp10_ = tf.get_variable("sig_p10",shape=[1,n_size_1[1]], dtype=tf.float32,initializer=tf.initializers.random_normal(mean=-5.0,stddev=0.1,seed=24))
    sigp10 = tf.exp(sigp10_)

    sigp11_ = tf.get_variable("sig_p11",shape=[1,n_size_2[1]], dtype=tf.float32,initializer=tf.initializers.random_normal(mean=-5.0,stddev=0.1,seed=25))
    sigp11 = tf.exp(sigp11_)

    mu_p00 = tf.get_variable("mu_p00",shape=[1,n_size_1[0]], dtype=tf.float32,initializer=tf.initializers.random_normal(mean=0.0,stddev=0.1,seed=26))
    mu_p01 = tf.get_variable("mu_p01",shape=[1,n_size_2[0]], dtype=tf.float32,initializer=tf.initializers.random_normal(mean=0.0,stddev=0.1,seed=27))
    mu_p10 = tf.get_variable("mu_p10",shape=[1,n_size_1[1]], dtype=tf.float32,initializer=tf.initializers.random_normal(mean=0.0,stddev=0.1,seed=28))
    mu_p11 = tf.get_variable("mu_p11",shape=[1,n_size_2[1]], dtype=tf.float32,initializer=tf.initializers.random_normal(mean=0.0,stddev=0.1,seed=29))
    
      
    scale = tf.placeholder(tf.float32, shape=())

    n_dta00 = tf.placeholder(tf.float32,[P[0],n_size_1[0]],name='data_n00')
    n_dta01 = tf.placeholder(tf.float32,[P[0],n_size_2[0]],name='data_n01')
    n_dta10 = tf.placeholder(tf.float32,[P[0],n_size_1[1]],name='data_n10')
    n_dta11 = tf.placeholder(tf.float32,[P[0],n_size_2[1]],name='data_n11')

    n_dta_test = tf.placeholder(tf.float32,[P[0],n_test],name='data_n_test')

    n_dta_valid = tf.placeholder(tf.float32,[P[0],n_size_1[1]+n_size_2[1]],name='data_n_valid')


    psi_sample = tf.squeeze(sample_hyper(noise_dim,K,psi_dim))

    mu_phi = tf.reshape(tf.slice(psi_sample,[0,0],[-1,psi_dim_phi]),shape=[K,P[0],V]) #K*P*V  
    mu_theta00 = tf.slice(psi_sample,[0,psi_dim_phi],[-1,psi_dim_theta_each])
    mu_theta01 = tf.slice(psi_sample,[0,psi_dim_phi+psi_dim_theta_each],[-1,psi_dim_theta_each])
    mu_theta10 = tf.slice(psi_sample,[0,psi_dim_phi+2*psi_dim_theta_each],[-1,psi_dim_theta_each])
    mu_theta11 = tf.slice(psi_sample,[0,psi_dim_phi+3*psi_dim_theta_each],[-1,psi_dim_theta_each])
    mu_u0 = tf.slice(psi_sample,[0,psi_dim_phi+4*psi_dim_theta_each],[-1,psi_dim_u_each])
    mu_u1 = tf.slice(psi_sample,[0,psi_dim_phi+4*psi_dim_theta_each+psi_dim_u_each],[-1,psi_dim_u_each])
    mu_nu = tf.slice(psi_sample,[0,psi_dim_phi+4*psi_dim_theta_each+2*psi_dim_u_each],[-1,psi_dim_nu])
    mu_b = tf.slice(psi_sample,[0,psi_dim_phi+4*psi_dim_theta_each+2*psi_dim_u_each+psi_dim_nu],[-1,psi_dim_b])
    mu_q = tf.slice(psi_sample,[0,psi_dim_phi+4*psi_dim_theta_each+2*psi_dim_u_each+psi_dim_nu+psi_dim_b],[-1,psi_dim_q])
    mu_c = tf.slice(psi_sample,[0,psi_dim_phi+4*psi_dim_theta_each+2*psi_dim_u_each+psi_dim_nu+psi_dim_b+psi_dim_q],[-1,psi_dim_c])
    mu_gamma = tf.slice(psi_sample,[0,psi_dim_phi+4*psi_dim_theta_each+2*psi_dim_u_each+psi_dim_nu+psi_dim_b+psi_dim_q+psi_dim_c],[-1,psi_dim_gamma])

    phi_sample = sample_logitn(mu_phi,sigdirichlet)#K*P*V
    mu_phi=tf.debugging.check_numerics(mu_phi,'mu_phi')
    sigdirichlet=tf.debugging.check_numerics(sigdirichlet,'sigdirichlet')
    phi_sample=tf.debugging.check_numerics(phi_sample,'phi_sample')
    theta00_sample = sample_ln(mu_theta00,sigtheta00) #K*V
    theta01_sample = sample_ln(mu_theta01,sigtheta01) #K*V
    theta10_sample = sample_ln(mu_theta10,sigtheta10) #K*V
    theta11_sample = sample_ln(mu_theta11,sigtheta11) #K*V

    u0_sample = sample_ln(mu_u0,sigu0) #K*V
    u1_sample = sample_ln(mu_u1,sigu1) #K*V
    b_sample = sample_ln(mu_b,sigb) #K*V
    nu_sample = sample_ln(mu_nu,signu) #K*Labs

    q_sample = sample_ln(mu_q,sigq) #K*Z

    c_sample = sample_ln(mu_c,sigc) #K*1
    gamma_sample = sample_ln(mu_gamma,siggamma) #K*1

    p00_sample = sample_logitn(tf.tile(mu_p00,[K,1]),sigp00) #K*n
    p01_sample = sample_logitn(tf.tile(mu_p01,[K,1]),sigp01) #K*n
    p10_sample = sample_logitn(tf.tile(mu_p10,[K,1]),sigp10) #K*n
    p11_sample = sample_logitn(tf.tile(mu_p11,[K,1]),sigp11) #K*n


    term1_H = tf.reduce_sum(log_lognormal(tf.expand_dims(theta00_sample,axis=0),tf.expand_dims(mu_theta00,axis=1),tf.expand_dims(sigtheta00,axis=1)),axis=2)+\
        tf.reduce_sum(log_lognormal(tf.expand_dims(theta01_sample,axis=0),tf.expand_dims(mu_theta01,axis=1),tf.expand_dims(sigtheta01,axis=1)),axis=2)+\
        tf.reduce_sum(log_lognormal(tf.expand_dims(theta10_sample,axis=0),tf.expand_dims(mu_theta10,axis=1),tf.expand_dims(sigtheta10,axis=1)),axis=2)+\
        tf.reduce_sum(log_lognormal(tf.expand_dims(theta11_sample,axis=0),tf.expand_dims(mu_theta11,axis=1),tf.expand_dims(sigtheta11,axis=1)),axis=2)+\
        tf.reduce_sum(log_lognormal(tf.expand_dims(u0_sample,axis=0),tf.expand_dims(mu_u0,axis=1),tf.expand_dims(sigu0,axis=1)),axis=2)+\
        tf.reduce_sum(log_lognormal(tf.expand_dims(u1_sample,axis=0),tf.expand_dims(mu_u1,axis=1),tf.expand_dims(sigu1,axis=1)),axis=2)+\
        tf.reduce_sum(log_lognormal(tf.expand_dims(b_sample,axis=0),tf.expand_dims(mu_b,axis=1),tf.expand_dims(sigb,axis=1)),axis=2)+\
        tf.reduce_sum(log_lognormal(tf.expand_dims(nu_sample,axis=0),tf.expand_dims(mu_nu,axis=1),tf.expand_dims(signu,axis=1)),axis=2)+\
        tf.reduce_sum(log_lognormal(tf.expand_dims(q_sample,axis=0),tf.expand_dims(mu_q,axis=1),tf.expand_dims(sigq,axis=1)),axis=2)+\
        tf.reduce_sum(log_lognormal(tf.expand_dims(c_sample,axis=0),tf.expand_dims(mu_c,axis=1),tf.expand_dims(sigc,axis=1)),axis=2)+\
        tf.reduce_sum(log_lognormal(tf.expand_dims(gamma_sample,axis=0),tf.expand_dims(mu_gamma,axis=1),tf.expand_dims(siggamma,axis=1)),axis=2)+\
        tf.reduce_sum(log_logitn(tf.expand_dims(p00_sample,axis=0),tf.expand_dims(mu_p00,axis=1),tf.expand_dims(sigp00,axis=1)),axis=2)+\
        tf.reduce_sum(log_logitn(tf.expand_dims(p01_sample,axis=0),tf.expand_dims(mu_p01,axis=1),tf.expand_dims(sigp01,axis=1)),axis=2)+\
        tf.reduce_sum(log_logitn(tf.expand_dims(p10_sample,axis=0),tf.expand_dims(mu_p10,axis=1),tf.expand_dims(sigp10,axis=1)),axis=2)+\
        tf.reduce_sum(log_logitn(tf.expand_dims(p11_sample,axis=0),tf.expand_dims(mu_p11,axis=1),tf.expand_dims(sigp11,axis=1)),axis=2) 

    term1_H=tf.debugging.check_numerics(term1_H,'1st H term')
    term2_H = tf.reduce_sum(tf.reduce_sum(log_logitn2(tf.expand_dims(phi_sample,axis=0),tf.expand_dims(mu_phi,axis=1),tf.expand_dims(sigdirichlet,axis=1)),axis=3),axis=2)

    term2_H=tf.debugging.check_numerics(term2_H,'2nd H term')
    log_H = tf.transpose(tf.reduce_logsumexp(term1_H+term2_H,axis=0,keepdims=True))-\
        tf.log(tf.cast(K,tf.float32)+1.0)  
    log_H=tf.debugging.check_numerics(log_H,'log H')

    pt1 = ((alpha0 - 1.0)*tf.log(gamma_sample+regul_)) - (beta0*gamma_sample)
    pt2 = ((a0 - 1.0)*tf.log(c_sample+regul_)) - (d0*c_sample)
    pt3 = tf.reduce_sum(((w0 - 1.0)*tf.log(q_sample+regul_)) - (u0*q_sample),axis=-1,keepdims=True)
    pt4 = tf.reduce_sum(((e0 - 1.0)*tf.log(nu_sample+regul_)) - (f0*nu_sample),axis=-1,keepdims=True)
    pt5 = tf.reduce_sum(((gamma_sample/V - 1.0)*tf.log(b_sample+regul_)) - (c_sample*b_sample),axis=-1,keepdims=True)
    pt6 = gamma_sample*tf.log(c_sample)-V*tf.lgamma(gamma_sample/V)
    pt7 = tf.reduce_sum(((b_sample - 1.0)*tf.log(u0_sample+regul_)) - (tf.slice(q_sample,[0,0],[K,1])*u0_sample),axis=-1,keepdims=True)
    pt8 = tf.reduce_sum(((b_sample - 1.0)*tf.log(u1_sample+regul_)) - (tf.slice(q_sample,[0,1],[K,1])*u1_sample),axis=-1,keepdims=True)
    pt9 = tf.reduce_sum(b_sample,axis=-1,keepdims=True)*tf.reduce_sum(tf.log(q_sample+regul_),axis=-1,keepdims=True)
    pt10= -dom_size*tf.reduce_sum(tf.lgamma(b_sample),axis=-1,keepdims=True)
    pt11= tf.reduce_sum(((u0_sample - 1.0)*tf.log(theta00_sample+regul_)) - (tf.slice(nu_sample,[0,0],[K,1])*theta00_sample),axis=-1,keepdims=True)
    pt12= tf.reduce_sum(((u0_sample - 1.0)*tf.log(theta01_sample+regul_)) - (tf.slice(nu_sample,[0,1],[K,1])*theta01_sample),axis=-1,keepdims=True)
    pt13= tf.reduce_sum(((u1_sample - 1.0)*tf.log(theta10_sample+regul_)) - (tf.slice(nu_sample,[0,0],[K,1])*theta10_sample),axis=-1,keepdims=True)
    pt14= tf.reduce_sum(((u1_sample - 1.0)*tf.log(theta11_sample+regul_)) - (tf.slice(nu_sample,[0,1],[K,1])*theta11_sample),axis=-1,keepdims=True)
    pt15= -n_labs*tf.reduce_sum(tf.lgamma(u0_sample),axis=-1,keepdims=True)-n_labs*tf.reduce_sum(tf.lgamma(u1_sample),axis=-1,keepdims=True)
    pt16= (tf.reduce_sum(tf.log(nu_sample+regul_),axis=-1,keepdims=True))*(tf.reduce_sum(u0_sample,axis=-1,keepdims=True)+tf.reduce_sum(u1_sample,axis=-1,keepdims=True))
    pt17= tf.reduce_sum(tf.reduce_sum(log_logitn2(phi_sample,tf.expand_dims(mu_phi_prior,axis=0),tf.expand_dims(sigdirichlet_prior,axis=0)),axis=-1),axis=-1,keepdims=True)
    pt18 = tf.reduce_sum((g0 - 1.0)*tf.log(p00_sample+regul_) + (h0 - 1.0)*tf.log(1.0-p00_sample+regul_),axis=-1,keepdims=True)
    pt19 = tf.reduce_sum((g0 - 1.0)*tf.log(p01_sample+regul_) + (h0 - 1.0)*tf.log(1.0-p01_sample+regul_),axis=-1,keepdims=True)
    pt20 = tf.reduce_sum((g0 - 1.0)*tf.log(p10_sample+regul_) + (h0 - 1.0)*tf.log(1.0-p10_sample+regul_),axis=-1,keepdims=True)
    pt21 = tf.reduce_sum((g0 - 1.0)*tf.log(p11_sample+regul_) + (h0 - 1.0)*tf.log(1.0-p11_sample+regul_),axis=-1,keepdims=True)
 
    log_P_prior = pt1+ pt2+ pt3+ pt4+ pt5+ pt6+ pt7+ pt8+ pt9 + pt10+ pt11+ pt12+ pt13+ pt14+ pt15+ pt16 + pt17 +pt18+pt19+pt20+pt21
 
    log_P_prior=tf.debugging.check_numerics(log_P_prior,'log P prior')
    N_train_P00 = tf.expand_dims(n_dta00,axis=0)#1*P*N
    N_train_P01 = tf.expand_dims(n_dta01,axis=0)#1*P*N
    N_train_P10 = tf.expand_dims(n_dta10,axis=0)#1*P*N
    N_train_P11 = tf.expand_dims(n_dta11,axis=0)#1*P*N

    thetaphi00_sample = tf.einsum('kpv,kv->kp',phi_sample,theta00_sample)
    thetaphi01_sample = tf.einsum('kpv,kv->kp',phi_sample,theta01_sample)
    thetaphi10_sample = tf.einsum('kpv,kv->kp',phi_sample,theta10_sample)
    thetaphi11_sample = tf.einsum('kpv,kv->kp',phi_sample,theta11_sample)

    nb_term_test10 = tf.lgamma(n_dta_test+tf.transpose(tf.slice(thetaphi10_sample,[0,0],[1,P[0]]))) - tf.lgamma(tf.transpose(tf.slice(thetaphi10_sample,[0,0],[1,P[0]])))
    nb_term_test11 = tf.lgamma(n_dta_test+tf.transpose(tf.slice(thetaphi11_sample,[0,0],[1,P[0]]))) - tf.lgamma(tf.transpose(tf.slice(thetaphi11_sample,[0,0],[1,P[0]])))

    nb_term_test10v = tf.lgamma(tf.expand_dims(n_dta_test,axis=-1)+tf.expand_dims(tf.transpose(thetaphi10_sample),axis=1)) - tf.expand_dims(tf.lgamma(tf.transpose(thetaphi10_sample)),axis=1)
    nb_term_test11v = tf.lgamma(tf.expand_dims(n_dta_test,axis=-1)+tf.expand_dims(tf.transpose(thetaphi11_sample),axis=1)) - tf.expand_dims(tf.lgamma(tf.transpose(thetaphi11_sample)),axis=1)

    nb_term_valid10 = tf.lgamma(tf.expand_dims(n_dta_valid,axis=-1)+tf.expand_dims(tf.transpose(thetaphi10_sample),axis=1)) - tf.expand_dims(tf.lgamma(tf.transpose(thetaphi10_sample)),axis=1)
    nb_term_valid11 = tf.lgamma(tf.expand_dims(n_dta_valid,axis=-1)+tf.expand_dims(tf.transpose(thetaphi11_sample),axis=1)) - tf.expand_dims(tf.lgamma(tf.transpose(thetaphi11_sample)),axis=1)

    p00_sample_P=tf.expand_dims(p00_sample,axis=1)#K*1*n
    p01_sample_P=tf.expand_dims(p01_sample,axis=1)
    p10_sample_P=tf.expand_dims(p10_sample,axis=1)
    p11_sample_P=tf.expand_dims(p11_sample,axis=1)

    log_P = tf.reduce_sum(tf.reduce_sum(tf.lgamma(tf.add(N_train_P00,tf.expand_dims(thetaphi00_sample,axis=-1))),axis=2),axis=1,keepdims=True)+\
        tf.reduce_sum(tf.reduce_sum(tf.lgamma(tf.add(N_train_P01,tf.expand_dims(thetaphi01_sample,axis=-1))),axis=2),axis=1,keepdims=True)+\
        tf.reduce_sum(tf.reduce_sum(tf.lgamma(tf.add(N_train_P10,tf.expand_dims(thetaphi10_sample,axis=-1))),axis=2),axis=1,keepdims=True)+\
        tf.reduce_sum(tf.reduce_sum(tf.lgamma(tf.add(N_train_P11,tf.expand_dims(thetaphi11_sample,axis=-1))),axis=2),axis=1,keepdims=True)\
        -n_size_1[0]*tf.reduce_sum(tf.lgamma(thetaphi00_sample),axis=1,keepdims=True)-n_size_2[0]*tf.reduce_sum(tf.lgamma(thetaphi01_sample),axis=1,keepdims=True)\
        -n_size_1[1]*tf.reduce_sum(tf.lgamma(thetaphi10_sample),axis=1,keepdims=True)-n_size_2[1]*tf.reduce_sum(tf.lgamma(thetaphi11_sample),axis=1,keepdims=True)+\
        tf.reduce_sum(tf.reduce_sum(tf.multiply(N_train_P00,tf.log(p00_sample_P+regul_)),axis=2),axis=1,keepdims=True)+\
        tf.reduce_sum(tf.reduce_sum(tf.multiply(N_train_P01,tf.log(p01_sample_P+regul_)),axis=2),axis=1,keepdims=True)+\
        tf.reduce_sum(tf.reduce_sum(tf.multiply(N_train_P10,tf.log(p10_sample_P+regul_)),axis=2),axis=1,keepdims=True)+\
        tf.reduce_sum(tf.reduce_sum(tf.multiply(N_train_P11,tf.log(p11_sample_P+regul_)),axis=2),axis=1,keepdims=True)+\
        tf.reduce_sum(tf.reduce_sum(tf.multiply(tf.expand_dims(thetaphi00_sample,axis=-1),tf.log(1.0-p00_sample_P+regul_)),axis=2),axis=1,keepdims=True)+\
        tf.reduce_sum(tf.reduce_sum(tf.multiply(tf.expand_dims(thetaphi01_sample,axis=-1),tf.log(1.0-p01_sample_P+regul_)),axis=2),axis=1,keepdims=True)+\
        tf.reduce_sum(tf.reduce_sum(tf.multiply(tf.expand_dims(thetaphi10_sample,axis=-1),tf.log(1.0-p10_sample_P+regul_)),axis=2),axis=1,keepdims=True)+\
        tf.reduce_sum(tf.reduce_sum(tf.multiply(tf.expand_dims(thetaphi11_sample,axis=-1),tf.log(1.0-p11_sample_P+regul_)),axis=2),axis=1,keepdims=True)      
    log_P=tf.debugging.check_numerics(log_P,'log P')
 
    #Prior Loss
    loss_prior = tf.reduce_sum(tf.reduce_sum(tf.abs(tf.slice(phi_sample,[0,indices_cnst[0][0],0],[K,1,V]) - tf.slice(phi_sample,[0,indices_cnst[1][0],0],[K,1,V])),axis=2),axis=1,keepdims=True)
    for cnst_cnt in range(cnst_num-1):
        loss_prior = loss_prior + tf.reduce_sum(tf.reduce_sum(tf.abs(tf.slice(phi_sample,[0,indices_cnst[0][cnst_cnt+1],0],[K,1,V]) - tf.slice(phi_sample,[0,indices_cnst[1][cnst_cnt+1],0],[K,1,V])),axis=2),axis=1,keepdims=True)

    loss = tf.reduce_mean(scale*(log_H - log_P - log_P_prior + 1.0*loss_prior))




    nn_var = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='hyper_q')
    lr=tf.constant(lr_N)


    train_op1 = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss,var_list=nn_var)

    lr2=tf.constant(lr_V)

    train_op2 = tf.train.GradientDescentOptimizer(learning_rate=lr2).minimize(loss,var_list=[sigdirichlet_,sigtheta_00,sigtheta_01,\
        sigtheta_10,sigtheta_11,sigu_0,sigu_1,signu_,sigb_,sigq_,sigc_,siggamma_,mu_p00,mu_p01,mu_p10,mu_p11,\
            sigp00_,sigp01_,sigp10_,sigp11_])


    init_op=tf.global_variables_initializer()

#%%

    sess=tf.InteractiveSession()

    sess.run(init_op)

    record = []
    
    opt_iters_V=np.floor(opt_iters*0.9)
    opt_iters_red=np.floor(opt_iters*0.33)

    best_err_1 = 1.0
    best_err_2 = 1.0
    phitheta10_test_tmp_bestval1=[]
    phitheta11_test_tmp_bestval1=[]
    nb_term_test10_tmp_v_best_val1=[]
    nb_term_test11_tmp_v_best_val1=[]
    phitheta10_test_tmp_bestval2=[]
    phitheta11_test_tmp_bestval2=[]
    nb_term_test10_tmp_v_best_val2=[]
    nb_term_test11_tmp_v_best_val2=[]
    for i in range(opt_iters):
        _,cost=sess.run([train_op1,loss],{n_dta00:N_training_1[0],n_dta01:N_training_2[0],n_dta10:N_training_1[1],n_dta11:N_training_2[1],n_dta_test:N_test,n_dta_valid:N_train[1],lr:lr_N*(0.7**(i/opt_iters_red)),scale:min(1.0e-0,1.0)})

        if i<opt_iters_V:
            if (lr_V/lr_N) < 0.01:
                for dum_cnt in range(3):
                    _,cost_n,=sess.run([train_op2,loss],{n_dta00:N_training_1[0],n_dta01:N_training_2[0],n_dta10:N_training_1[1],n_dta11:N_training_2[1],n_dta_test:N_test,n_dta_valid:N_train[1],lr2:lr_V*(0.9**(i/opt_iters_red)),scale:min(1.0e-0,1.0)})
            else:
                _,cost_n,=sess.run([train_op2,loss],{n_dta00:N_training_1[0],n_dta01:N_training_2[0],n_dta10:N_training_1[1],n_dta11:N_training_2[1],n_dta_test:N_test,n_dta_valid:N_train[1],lr2:lr_V*(0.9**(i/opt_iters_red)),scale:min(1.0e-0,1.0)})
   
        record.append(cost)
        if i%1 == 0:
            print("iter:", '%04d' % (i+1), "cost=", np.mean(record),',', np.std(record),"cost_N=",cost,"cost_V=",cost_n)
            record = []
            phitheta10_test_tmp,phitheta11_test_tmp,nb_term_test10_tmp_v,nb_term_test11_tmp_v,phi_test_tmp,nb_term_valid10_tmp,nb_term_valid11_tmp=sess.run([thetaphi10_sample,thetaphi11_sample,nb_term_test10v,nb_term_test11v,phi_sample,nb_term_valid10,nb_term_valid11],{n_dta00:N_training_1[0],n_dta01:N_training_2[0],n_dta10:N_training_1[1],n_dta11:N_training_2[1],n_dta_test:N_test,n_dta_valid:N_train[1]})
            tmp_err_1,tmp_err_2=obc_valid_test_error(N_test,N_train[1],X_test,X_train[1],class_prior,phitheta10_test_tmp,phitheta11_test_tmp,nb_term_test10_tmp_v,nb_term_test11_tmp_v,nb_term_valid10_tmp,nb_term_valid11_tmp)
            if tmp_err_1 < best_err_1:
                best_err_1 = tmp_err_1
                phitheta10_test_tmp_bestval1=[]
                phitheta11_test_tmp_bestval1=[]
                nb_term_test10_tmp_v_best_val1=[]
                nb_term_test11_tmp_v_best_val1=[]
                phitheta10_test_tmp_bestval1.append(phitheta10_test_tmp)
                phitheta11_test_tmp_bestval1.append(phitheta11_test_tmp)
                nb_term_test10_tmp_v_best_val1.append(nb_term_test10_tmp_v)
                nb_term_test11_tmp_v_best_val1.append(nb_term_test11_tmp_v)
            if tmp_err_2 < best_err_2:
                best_err_2 = tmp_err_2
                phitheta10_test_tmp_bestval2=[]
                phitheta11_test_tmp_bestval2=[]
                nb_term_test10_tmp_v_best_val2=[]
                nb_term_test11_tmp_v_best_val2=[]
                phitheta10_test_tmp_bestval2.append(phitheta10_test_tmp)
                phitheta11_test_tmp_bestval2.append(phitheta11_test_tmp)
                nb_term_test10_tmp_v_best_val2.append(nb_term_test10_tmp_v)
                nb_term_test11_tmp_v_best_val2.append(nb_term_test11_tmp_v)
            if tmp_err_1 == best_err_1:
                phitheta10_test_tmp_bestval1.append(phitheta10_test_tmp)
                phitheta11_test_tmp_bestval1.append(phitheta11_test_tmp)
                nb_term_test10_tmp_v_best_val1.append(nb_term_test10_tmp_v)
                nb_term_test11_tmp_v_best_val1.append(nb_term_test11_tmp_v)
            if tmp_err_2 == best_err_2:
                phitheta10_test_tmp_bestval2.append(phitheta10_test_tmp)
                phitheta11_test_tmp_bestval2.append(phitheta11_test_tmp)
                nb_term_test10_tmp_v_best_val2.append(nb_term_test10_tmp_v)
                nb_term_test11_tmp_v_best_val2.append(nb_term_test11_tmp_v)

    phitheta10_test_tmp_bestval1=np.concatenate(phitheta10_test_tmp_bestval1,axis=0)
    phitheta11_test_tmp_bestval1=np.concatenate(phitheta11_test_tmp_bestval1,axis=0)
    phitheta10_test_tmp_bestval2=np.concatenate(phitheta10_test_tmp_bestval2,axis=0)
    phitheta11_test_tmp_bestval2=np.concatenate(phitheta11_test_tmp_bestval2,axis=0)
    nb_term_test10_tmp_v_best_val1=np.concatenate(nb_term_test10_tmp_v_best_val1,axis=2)
    nb_term_test11_tmp_v_best_val1=np.concatenate(nb_term_test11_tmp_v_best_val1,axis=2)
    nb_term_test10_tmp_v_best_val2=np.concatenate(nb_term_test10_tmp_v_best_val2,axis=2)
    nb_term_test11_tmp_v_best_val2=np.concatenate(nb_term_test11_tmp_v_best_val2,axis=2)
    test_err_1,test_err_2=obc_test_error(N_test,N_train[1],X_test,X_train[1],class_prior,phitheta10_test_tmp_bestval1,phitheta11_test_tmp_bestval1,nb_term_test10_tmp_v_best_val1,nb_term_test11_tmp_v_best_val1)
    test_err_1,test_err_2=obc_test_error(N_test,N_train[1],X_test,X_train[1],class_prior,phitheta10_test_tmp_bestval2,phitheta11_test_tmp_bestval2,nb_term_test10_tmp_v_best_val2,nb_term_test11_tmp_v_best_val2)
 

    data_dic = {"obc_test_error1" : test_err_1.tolist(),"obc_test_error2" : test_err_2.tolist()} 
    f_name_1 = file_name + '.mat'
    f_name_2 = file_name + '.json'
    savemat(f_name_1,data_dic)
    with open(f_name_2, 'w') as fp:
        json.dump(data_dic, fp)
    return 0
