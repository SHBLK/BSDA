
import tensorflow as tf
import numpy as np


regul_=0.0#1e-12
regul_2=1e-12#0.0

def lognormal(z,mu,sigma):
    pdf = 1/(sigma*z)*tf.exp(-0.5*tf.square(tf.log(z)-mu)/tf.square(sigma))
    return pdf

def log_normal(z,mu,sigma):
    pdf = -tf.log(sigma+regul_)-0.5*tf.square(z-mu)/tf.square(sigma)
    return pdf


def log_lognormal(z,mu,sigma):
    pdf = (-tf.log(sigma+regul_)-tf.log(z+regul_))+(-0.5*tf.square(tf.log(z+ regul_)-mu)/tf.square(sigma)) 
    return pdf


def sample_ln(mu,sigma):
    eps = tf.random_normal(shape=tf.shape(mu))
    z=tf.exp(mu+eps*sigma)
    return z

def logitnormal(z,mu,sigma):
    logit = tf.log(z/(1-z))
    term1 = 1/(z*(1-z))
    term2 = 1/(sigma)*tf.exp(-0.5*tf.square(logit-mu)/tf.square(sigma))
    pdf = term1*term2
    return pdf

def log_logitn(z,mu,sigma):
    logit = tf.log(z+regul_)-tf.log(1-z+regul_)
    pdf = (-tf.log(sigma+regul_)-tf.log(z+regul_)-tf.log(1.0-z+regul_)) + (-0.5*tf.square(logit-mu)/tf.square(sigma))
    return pdf

def log_logitn2(z,mu,sigma):
    logit = tf.log(z+regul_2)-tf.log(1-z+regul_2)
    pdf = (-tf.log(sigma+regul_2)-tf.log(z+regul_2)-tf.log(1.0-z+regul_2)) + (-0.5*tf.square(logit-mu)/(tf.square(sigma)+regul_2))
    return pdf
    
def sample_logitn(mu,sigma):
    eps = tf.random_normal(shape=tf.shape(mu))
    z = mu+eps*sigma
    return tf.exp(z)/(1+tf.exp(z))

def sample_n(mu,sigma):

    eps = tf.random_normal(shape=tf.shape(mu))
    z = mu+eps*sigma#tf.matmul(eps,sigma)   
    return z

def sample_n_e(mu,sigma):

    eps = tf.random_normal(shape=tf.shape(mu))
    z = mu+tf.einsum('kpv,pmv->kpm',eps,sigma)   
    return z

def log_sum_exp(x,axis=0):
    #x = np.rollaxis(x,axis)
    max_x=x.max(axis=0)
    out = np.log(np.sum(np.exp(x - max_x), axis=0))
    out += max_x
    return out

def give_obc_error(log_p,c_prior,true_labs):
    true_labs = true_labs - 1
    true_labs = true_labs.astype(np.bool)
    N = len(true_labs)
    pred_labels=(np.log(c_prior[0])+log_p[0])<(np.log(c_prior[1])+log_p[1])
    TP = np.sum(pred_labels & true_labs)
    FP = np.sum(pred_labels) - TP
    TN = np.sum((1-pred_labels) & (1-true_labs))
    FN = np.sum((1-pred_labels)) - TN
    return TP,FP,TN,FN,float(FP+FN)/float(N)

def give_error(p_pred,true_labs):
    true_labs = true_labs - 1
    true_labs = true_labs.astype(np.bool)
    N = len(true_labs)
    pred_labels=p_pred>=0.5
    TP = np.sum(pred_labels & true_labs)
    FP = np.sum(pred_labels) - TP
    TN = np.sum((1-pred_labels) & (1-true_labs))
    FN = np.sum((1-pred_labels)) - TN
    return TP,FP,TN,FN,float(FP+FN)/float(N)




