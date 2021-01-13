import numpy as np
import random
import pandas as pd
from sklearn.metrics import roc_curve, roc_auc_score, average_precision_score

import matplotlib
matplotlib.use('AGG')

from matplotlib import pyplot as plt
from Bio import SeqIO

from .dp_z import seq2int_cy, generate_kmer_inx, DP_Z_cy
from .LL_avx import nLL

np_float_t = np.float32
#np_float_t = np.float64

def auc_evaluate(param, plus, bg, core_length, kmer_inx):

    z_plus = np.zeros(len(plus), dtype=np_float_t)
    z_bg = np.zeros(len(bg), dtype=np_float_t)
    
    seq_pos = [seq2int_cy('A' + x, core_length, kmer_inx) for x in plus]
    seq_bg = [seq2int_cy('A' + x, core_length, kmer_inx) for x in bg]
    
    n_pos = 3
    #exp parameters to make sure they are positive
    args = param.copy()
    args[-n_pos:-1] = np.exp(args[-n_pos:-1])
    exp_p = np.exp(args[-1])
    args[-1] = exp_p/(1+exp_p)
        
    for i, x in enumerate(seq_pos):
        z_plus[i], _ = DP_Z_cy(args, x, core_length)
    
    for i, x in enumerate(seq_bg):
        z_bg[i], _ = DP_Z_cy(args, x, core_length)
        
    y_true = np.append(np.ones(len(plus)), np.zeros(len(bg)))
    y_score = np.append(z_plus, z_bg)
    
    fpr_grd, tpr_grd, _ = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)
    
    return fpr_grd, tpr_grd, auc

def predict(test_sequences, theta, core_length, kmer_inx):

    z = np.zeros(len(test_sequences), dtype=np_float_t)
    
    seq_int = [seq2int_cy('A' + x, core_length, kmer_inx) for x in test_sequences]
    
    n_pos = 3
    #exp parameters to make sure they are positive
    args = theta.copy()
    args[-n_pos:-1] = np.exp(args[-n_pos:-1])
    exp_p = np.exp(args[-1])
    args[-1] = exp_p/(1+exp_p)
        
    for i, x in enumerate(seq_int):
        z[i], _ = DP_Z_cy(args, x, core_length)
        
    return z

def evaluate_test_set(param, plus, bg, core_length, kmer_inx):
    z_ps = np.zeros(len(plus), dtype=np_float_t)
    z_bg = np.zeros(len(bg), dtype=np_float_t)
    
    seq_ps = [seq2int_cy('A' + x, core_length, kmer_inx) for x in plus]
    seq_bg = [seq2int_cy('A' + x, core_length, kmer_inx) for x in bg]
    
    n_pos = 3
    #exp parameters to make sure they are positive
    args = param.copy()
    args[-n_pos:-1] = np.exp(args[-n_pos:-1])
    exp_p = np.exp(args[-1])
    args[-1] = exp_p/(1+exp_p)
        
    for i, x in enumerate(seq_ps):
        z_ps[i], _ = DP_Z_cy(args, x, core_length)
    
    for i, x in enumerate(seq_bg):
        z_bg[i], _ = DP_Z_cy(args, x, core_length)
        
    y_true = np.append(np.ones(len(plus)), np.zeros(len(bg)))
    y_score = np.append(z_ps, z_bg)
    
    auc = roc_auc_score(y_true, y_score)
    ap = average_precision_score(y_true, y_score)

    
    return auc, ap

def partition (list_in, n):
    new_list = list_in.copy()
    random.shuffle(new_list)
    return [new_list[i::n] for i in range(n)]


def plt_performance(plus, bg, param_history, core_length, kmer_inx, file_name, evaluate_after, final_plot=True, ll_hist=None):

    inx_kmer = dict((v,k) for k,v in kmer_inx.items())
        
    fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(12,3))
    
    theta = param_history[-1]
    xx = np.arange(1, len(param_history)*evaluate_after+1, evaluate_after)

    # get current motif =======================
    
    core1 = {}
    for i in range(len(kmer_inx)):
        core1[inx_kmer[i]] = theta[i]

    core1 = pd.Series(core1).sort_values(ascending=True)
    
    core2 = {}
    for i in range(len(kmer_inx)):
        core2[inx_kmer[i]] = theta[i+len(kmer_inx)]
    core2 = pd.Series(core2).sort_values(ascending=True)
    
    r = np.exp(theta[-2])
    p = 1/(1+np.exp(-theta[-1]))
    

    # set plot title ===========================

    fig.suptitle(f'{core1.index[0]}({core1.values[0]:.2f}) -- r={r:.1f},p={p:.1f} -- {core2.index[0]}({core2.values[0]:.2f})\n' +
    f'{core1.index[1]}({core1.values[1]:.2f}) ----------------- {core2.index[1]}({core2.values[1]:.2f})\n' +
    f'{core1.index[2]}({core1.values[2]:.2f}) ----------------- {core2.index[2]}({core2.values[2]:.2f})\n' +
    f'variation index: {param_local_fluctuation(param_history):.2f}', horizontalalignment='left', fontsize=12, x=0.1, y=1.2)
        
    if final_plot:
        # AUC plots ================================
        fpr_t, tpr_t, auct = auc_evaluate(theta, plus, bg, core_length, kmer_inx)
            
        ax1.plot([0, 1], [0, 1], 'k--')
        ax1.plot(fpr_t, tpr_t, label=f'training (AUC={auct:.2f})')
        ax1.set_xlabel('False positive rate')
        ax1.set_ylabel('True positive rate')
        ax1.set_title('ROC curve')
        ax1.legend(loc='best')
    
    else:
        # LL history plots =========================
        ax1.plot(xx, ll_hist) 
        ax1.set_xlabel('iteration')
        ax1.set_ylabel('log-likelihood')



    # Plot binding energies of best bound k-mer per core ====
    for i in range(len(kmer_inx)):
        core1_hist = [arr[i] for arr in param_history]
        core2_hist = [arr[i + len(kmer_inx)] for arr in param_history]
        
        ax2.plot(xx, core1_hist, color='#c0c5d1', linewidth=1.5)
        ax2.plot(xx, core2_hist, color='#d1c0c3', linewidth=1.5)

    #plot best kmers for each core
    kmer1 = core1.index[0]
    kmer2 = core2.index[0]

    core1_hist = [arr[kmer_inx[kmer1]] for arr in param_history]
    core2_hist = [arr[kmer_inx[kmer2]+ len(kmer_inx)] for arr in param_history]
    
    ax2.plot(xx, core1_hist, color='#092e87', label='core1 %s E'%kmer1, ls='--')
    ax2.plot(xx, core2_hist, color='#870b1b', label='core2 %s E'%kmer2, ls='--')

    #plot second best kmers for each core
    if True:
        kmer1 = core1.index[1]
        kmer2 = core2.index[1]

        core1_hist = [arr[kmer_inx[kmer1]] for arr in param_history]
        core2_hist = [arr[kmer_inx[kmer2]+ len(kmer_inx)] for arr in param_history]
        
        ax2.plot(xx, core1_hist, color='#097d87', label='core1: %s energy'%kmer1, ls='--')
        ax2.plot(xx, core2_hist, color='#873f0b', label='core2: %s energy'%kmer2, ls='--')
    
    ax2.set_xlabel('iteration')
    ax2.legend(loc='upper right')
        
    # plot r, p, and SF ========================
    
    r_hist = [np.exp(arr[-2]) for arr in param_history]
    p_hist = [1/(1+np.exp(-arr[-1])) for arr in param_history]
    
    mean_hist = [((1-p)*r)/(p) for r,p in zip(r_hist, p_hist)]
    mode_hist = [max(0,int(((1-p)*(r-1))/p)) for r,p in zip(r_hist, p_hist)]

    ax3.set_xlabel('iteration')
    ax3.plot(xx, mean_hist, color='#092e87', label='distance mean')  
    ax3.plot(xx, mode_hist, color='#870b1b', label='distance mode')
    
    ax3.legend() 
    
    #================================================
    
    plt.savefig(f'{file_name}.pdf', bbox_inches='tight')
    plt.savefig(f'{file_name}.png', bbox_inches='tight', dpi=150)

    plt.close()
    


#returns a fluctuation score [0-1] max representing parameter variation in the last 5 iterations.

def param_local_fluctuation(param_history):
    
    last_params = list(param_history[-1])
    min_energy_inx = last_params.index(min(last_params[:-2]))
    energy_hist = [arr[min_energy_inx] for arr in param_history]
    
    r_hist = [np.exp(arr[-2]) for arr in param_history]
    p_hist = [np.exp(arr[-1]) for arr in param_history]
    
    #define the strech which is assigned as local
    #if less than 5 iterations --> return 1
    if len(param_history)<5:
        return 1
    else:
        loc_len = 5
       
    #max(arr)-min(arr) for the last 5 elements of this parameter in adam optimization
    local_variation=np.array([max(a)-min(a) for a in [energy_hist[-loc_len:], r_hist[-loc_len:], p_hist[-loc_len:]]])
        
    #return biggest ratio of local to absolute value
    return max(local_variation/(np.array([energy_hist[-1], r_hist[-1], p_hist[-1]])+1))



def optimize_adam(plus, bg, 
                  parameters,
                  core_length=3, 
                  var_thr=0.05, 
                  sequences_per_batch=100, 
                  max_iterations=1000, 
                  evaluate_after=None, 
                  no_cores=4,
                  save_files=True, 
                  file_name='bipartite'):

    #number of minibatches: number of positives/numbers per batch
    n_batch = int(len(plus)/sequences_per_batch)
    
    #adam parameters
    alpha = 0.01
    beta_1 = 0.9
    beta_2 = 0.999   
    epsilon = 1e-8

    #initialize the vector
    theta_0 = parameters  
    n_param = len(parameters)

    m_t = np.zeros(n_param, dtype=np_float_t) 
    v_t = np.zeros(n_param, dtype=np_float_t)  

    t = 0  #iterations
    f_t = 42 #initialize with random number
    epoch = 0
    
    min_iter = 20 #minimum rounds of parameter recording before checking for convergence

    param_history = []
    ll_history = []
    
    #if none assign an epoch evaluation scheme
    if evaluate_after is None:
        evaluate_after = int(len(plus)/sequences_per_batch)
    
    # create kmer-index 
    kmer_inx = generate_kmer_inx(core_length)
    
    while(True):
        
        #split data into several minibatches
        epoch += 1

        pos_batches = partition(plus, n_batch)
        bg_batches  = partition(bg, n_batch)
        
        #enumerate minibatches
        for i in range(n_batch):
            
            nll_obj = nLL(pos_batches[i],bg_batches[i], core_length, no_cores)

            t+=1

            #computes the gradient of the stochastic function
            f_t, g_t = nll_obj(theta_0) 

            #updates the moving averages of the gradient
            m_t = beta_1*m_t + (1-beta_1)*g_t 

            #updates the moving averages of the squared gradient
            v_t = beta_2*v_t + (1-beta_2)*(g_t*g_t) 

            #calculates the bias-corrected estimates
            m_cap = m_t/(1-(beta_1**t)) 

            #calculates the bias-corrected estimates
            v_cap = v_t/(1-(beta_2**t)) 

            #before updating parameters plot performance on validation set
            
            #every now and then: evaluate after $evaluate_after iterations
            if t%evaluate_after == 0:                
                #track parameter evolution
                param_history.append(theta_0)
                ll_history.append(-f_t)

                if t%(evaluate_after*4) == 0:
                    plt_performance(plus, bg, param_history, core_length, kmer_inx, file_name, evaluate_after, final_plot=False, ll_hist=ll_history)
                
                #minimum iterations before checking for convergence: min_iter                
                if len(param_history)>min_iter:

                    #stop when the parameters are stable (see param_local_fluctuation)
                    #OR stop when it took too long ($max_iterations)
                    variability_index = param_local_fluctuation(param_history)
                    if variability_index<var_thr or t>max_iterations:
                        if save_files:
                            plt_performance(plus, bg, param_history, core_length, kmer_inx, file_name, evaluate_after)
                            np.savetxt(fname=file_name +'.txt', X=np.append(theta_0,[f_t]))
                        return theta_0, g_t  

                    print(f'Variability index at iteration {t}/{max_iterations}: {variability_index:.3f}')             
                
            #updates the parameters by moving a step towards gradients
            theta_0 = theta_0 - (alpha*m_cap)/(np.sqrt(v_cap)+epsilon)     
            

def read_params(files):
    params = []
    for f in files:
        param = np.loadtxt(f)
        params.append(param)      
    return params