def auc_evaluate(param, plus, bg, core_length, kmer_inx):
    z_plus = np.zeros(len(plus))
    z_bg = np.zeros(len(bg))
    
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



def partition (list_in, n):
    new_list = list_in.copy()
    random.shuffle(new_list)
    return [new_list[i::n] for i in range(n)]


def plt_performance(plus, bg, plus_valid, bg_valid, param_history, core_length, kmer_inx):
    
    inx_kmer = dict((v,k) for k,v in kmer_inx.items())
        
    fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(12,3.5))
    
    theta = param_history[-1]
    x = np.arange(1, len(param_history)+1, 1)

    # get current motif =======================
    
    core1 = {}
    for i in range(len(kmer_inx)):
        core1[inx_kmer[i]] = theta[i]

    core1 = pd.Series(core1).sort_values(ascending=True)
    
    core2 = {}
    for i in range(len(kmer_inx)):
        core2[inx_kmer[i]] = theta[i+64]
    core2 = pd.Series(core2).sort_values(ascending=True)
    
    r = np.exp(theta[-2])
    p = 1/(1+np.exp(-theta[-1]))
    

    # set plot title ===========================

    fig.suptitle(f'{core1.index[0]}({core1.values[0]:.2f}) -- r={r:.1f},p={p:.1f} -- {core2.index[0]}({core2.values[0]:.2f})\n' +
    f'{core1.index[1]}({core1.values[1]:.2f}) ----------------- {core2.index[1]}({core2.values[1]:.2f})\n' +
    f'{core1.index[2]}({core1.values[2]:.2f}) ----------------- {core2.index[2]}({core2.values[2]:.2f})\n' +
    f'variation index: {param_local_fluctuation(param_history):.2f}', horizontalalignment='left', fontsize=12, x=0.1, y=1.2)
        
    # AUC plots ================================
    
    fpr_v, tpr_v, aucv = auc_evaluate(theta, plus_valid, bg_valid, core_length, kmer_inx)
    fpr_t, tpr_t, auct = auc_evaluate(theta, plus, bg, core_length, kmer_inx)
        
    ax1.plot([0, 1], [0, 1], 'k--')
    ax1.plot(fpr_v, tpr_v, label=f'validation (AUC={aucv:.2f})')
    ax1.plot(fpr_t, tpr_t, label=f'training (AUC={auct:.2f})')
    ax1.set_xlabel('False positive rate')
    ax1.set_ylabel('True positive rate')
    ax1.set_title('ROC curve')
    ax1.legend(loc='best')

    # Plot binding energies of best bound k-mer per core ====
    
    kmer1 = core1.index[0]
    kmer2 = core2.index[0]
    
    core1_hist = [arr[kmer_inx[kmer1]] for arr in param_history]
    core2_hist = [arr[kmer_inx[kmer2]+ len(kmer_inx)] for arr in param_history]
    
    ax2.plot(x, core1_hist, color='blue', label='core1 %s E'%kmer1)
    ax2.plot(x, core2_hist, color='red', label='core2 %s E'%kmer2)
    
    ax2.set_xlabel('n\'th iteration')
    ax2.legend()
        
    # plot r, p, and SF ========================
    
    sf_hist = [np.exp(arr[-3]) for arr in param_history]
    r_hist = [np.exp(arr[-2]) for arr in param_history]
    p_hist = [1/(1+np.exp(-arr[-1])) for arr in param_history]
    
    ax3.set_xlabel('n\'th iteration')
    ax3.plot(x, r_hist, color='blue', label='r')  
    #ax3.plot(x, p_hist, color='red', label='p')
    
    ax4 = ax3.twinx()  # instantiate a second axes that shares the same x-axis
    ax4.set_ylabel('p', color='red')  # we already handled the x-label with ax1
    ax4.plot(x, p_hist, color='red', label='p')
    ax4.tick_params(axis='y', labelcolor='red')    
    
    ax3.legend() 
    ax4.legend()
    
    #================================================
    
    plt.savefig('plots/'+ file_name +'.pdf', bbox_inches='tight')
    return auct, aucv
    


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



def optimize_adam(plus, bg, plus_valid, bg_valid, core_length=3, var_thr=0.05, 
                  sequences_per_batch=100, max_iterations=1000, evaluate_after=None, save_files=True):

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

    m_t = np.zeros(n_param) 
    v_t = np.zeros(n_param)  

    t = 0  #iterations
    f_t = 42 #initialize with random number
    epoch = 0
    
    min_iter = 20 #minimum rounds of parameter recording before checking for convergence

    #auc array tracks auc values
    param_history = []    
    
    #calculate the stretch of reduced performance
    reduction = 0
    
    #if none assign an epoch evaluation scheme
    if evaluate_after is None:
        evaluate_after = len(plus)
    
    # create kmer-index 
    kmer_inx = generate_kmer_inx(core_length)
    
    while(True):
        
        #split data into several minibatches
        epoch += 1

        pos_batches = partition(plus, n_batch)
        bg_batches  = partition(bg, n_batch)
        
        #enumerate minibatches
        for i in range(n_batch):
            
            nll_obj = nLL(pos_batches[i],bg_batches[i], core_length)

            t+=1

            f_prev = f_t

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
            
            #every now and then: evaluate after $evaluate_after sequences
            if (t*sequences_per_batch)%evaluate_after < sequences_per_batch: 
                
                #track parameter evolution
                param_history.append(theta_0)
                
                #minimum iterations before checking for convergence: min_iter                
                if len(param_history)>min_iter:

                    #stop when the parameters are stable (see param_local_fluctuation)
                    #OR stop when it took too long ($max_iterations)
                    
                    variability_index = param_local_fluctuation(param_history)
                    if variability_index<var_thr or t>max_iterations:
                        if save_files:
                            auct, aucv = plt_performance(plus, bg, plus_valid, bg_valid, param_history, core_length, kmer_inx)
                            np.savetxt(fname='param/'+ file_name +'.txt', X=np.append(theta_0,[f_t, auct, aucv]))
                        return theta_0, g_t               
                
            #updates the parameters by moving a step towards gradients
            theta_0_prev = theta_0 
            theta_0 = theta_0 - (alpha*m_cap)/(np.sqrt(v_cap)+epsilon)     
            
            


# ### Import fasta files

def parse_fastq(file_name):
    input_seq_iterator = SeqIO.parse(file_name, "fastq")
    return [str(record.seq) for record in input_seq_iterator]

def parse_fasta(file_name):
    input_seq_iterator = SeqIO.parse(file_name, "fasta")
    return [str(record.seq) for record in input_seq_iterator]

def parse_seq(file_name):
    with open(file_name,'r') as f:
        seq = [line.rstrip() for line in f]
    return seq