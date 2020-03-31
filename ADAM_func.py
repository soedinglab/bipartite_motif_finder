def auc_evaluate(param, plus, bg):
    z_plus = np.zeros(len(plus))
    z_bg = np.zeros(len(bg))
    
    seq_pos = [seq2int_cy('A' + x) for x in plus]
    seq_bg = [seq2int_cy('A' + x) for x in bg]
    
    n_pos = 3
    #exp parameters to make sure they are positive
    args = param.copy()
    args[-n_pos:] = np.exp(args[-n_pos:])
        
    for i, x in enumerate(seq_pos):
        z_plus[i], _ = DP_Z_cy(args, x)
    
    for i, x in enumerate(seq_bg):
        z_bg[i], _ = DP_Z_cy(args, x)
        
    y_true = np.append(np.ones(len(plus)), np.zeros(len(bg)))
    y_score = np.append(z_plus, z_bg)
    
    auc = roc_auc_score(y_true, y_score)
    
    return auc



def partition (list_in, n):
    random.shuffle(list_in)
    return [list_in[i::n] for i in range(n)]


def plt_performance(auc_validation, auc_train, param_history, theta):
        
    x = np.arange(1, len(auc_validation)+1, 1)
    
    fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(12,3.5))
    ax1.set_xlabel('10th iteration')
    ax1.set_ylabel('AUC')
    
    x_max = max(30, len(auc_validation))
    
    ax1.set_xlim(0,x_max)
    ax1.set_ylim(0.4,1)
    
    # get current motif ===========
    core1 = {}
    for i in range(len(kmer_inx)):
        core1[inx_kmer[i]] = theta[i]

    core1 = pd.Series(core1).sort_values(ascending=True)
    
    core2 = {}
    for i in range(len(kmer_inx)):
        core2[inx_kmer[i]] = theta[i+64]
    core2 = pd.Series(core2).sort_values(ascending=True)
    
    d = np.exp(theta[-2])
    sig = np.exp(theta[-1])
    #===============================================

    ax1.set_title('%s(%.2f) -- %.1f(%.1f) -- %s(%.2f)\n%s(%.2f) ----------------- %s(%.2f)\n%s(%.2f) ----------------- %s(%.2f)\ncurrent validation AUC: %.3f\nvariation index: %.2f'%(
        core1.index[0], core1.values[0],d,sig,core2.index[0], core2.values[0], 
        core1.index[1], core1.values[1], core2.index[1], core2.values[1], 
        core1.index[2], core1.values[2], core2.index[2], core2.values[2],
        auc_validation[-1],
        param_local_fluctuation(param_history)),
                loc='left')
    ax1.plot(x, auc_validation, color='blue', label='validation set ')  
    ax1.plot(x, auc_train, color='red', label='training set')
    ax1.legend()
    
    #Plot binding energies of one kmer per core ====
    
    core1_hist = [arr[kmer_inx[kmer1]] for arr in param_history]
    core2_hist = [arr[kmer_inx[kmer2]+ len(kmer_inx)] for arr in param_history]
    
    ax2.plot(x, core1_hist, color='blue', label='core1 %s E'%kmer1)
    ax2.plot(x, core2_hist, color='red', label='core2 %s E'%kmer2)
    ax3.set_xlabel('10th iteration')
    ax2.legend()
    ax2.set_xlim(0,x_max)
        
    # plot sigma, D, and SF ========================
    sf_hist = [np.exp(arr[-3]) for arr in param_history]
    D_hist = [np.exp(arr[-2]) for arr in param_history]
    sig_hist = [np.exp(arr[-1]) for arr in param_history]
    
    ax3.set_xlabel('10th iteration')
    ax3.plot(x, D_hist, color='blue', label='D')  
    ax3.plot(x, sig_hist, color='red', label='sigma')
    
    ax4 = ax3.twinx()  # instantiate a second axes that shares the same x-axis
    ax4.set_ylabel('sf', color='green')  # we already handled the x-label with ax1
    ax4.plot(x, sf_hist, color='green', label='sf')
    ax4.tick_params(axis='y', labelcolor='green')    
    
    ax3.legend()    
    ax3.set_xlim(0,x_max)
    
    #================================================
    
    plt.savefig('plots/'+ plot_name +'.pdf', bbox_inches='tight')
    
    clear_output(wait=True)
    plt.show()


#returns a fluctuation score [0-1] max representing parameter variation in the last 5 iterations.

def param_local_fluctuation(param_history):
    
    last_params = list(param_history[-1])
    min_energy_inx = last_params.index(min(last_params[:-3]))
    energy_hist = [np.exp(arr[min_energy_inx]) for arr in param_history]
    
    D_hist = [np.exp(arr[-2]) for arr in param_history]
    sig_hist = [np.exp(arr[-1]) for arr in param_history]
    
    #define the strech which is assigned as local
    #if less than 5 iterations --> return 1
    if len(param_history)<5:
        return 1
    else:
        loc_len = 5
       
    #max(arr)-min(arr) for the last 5 elements of this parameter in adam optimization
    local_variation=np.array([max(a)-min(a) for a in [energy_hist[-loc_len:], D_hist[-loc_len:], sig_hist[-loc_len:]]])
    
    #max(arr)-min(arr) for all parameter history in adam optimization
    #global_variation=np.array([max(a)-min(a) for a in [energy_hist, D_hist, sig_hist]])
        
    #return biggest ratio of local to absolute value
    return max(local_variation/(np.array([energy_hist[-1], D_hist[-1], sig_hist[-1]])+1))



def optimize_adam(plus, bg, plus_valid, bg_valid, red_thr=10, var_thr=0.05, 
                  sequences_per_batch=100, max_iterations=1000, evaluate_after=None):

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


    #auc array tracks auc values
    auc_validation = []
    auc_train = []
    param_history = []    
    
    #calculate the stretch of reduced performance
    reduction = 0
    
    #if none assign an epoch evaluation scheme
    if evaluate_after is None:
        evaluate_after = len(plus)
    
    while(True):
        
        #split data into several minibatches
        epoch += 1

        pos_batches = partition(plus, n_batch)
        bg_batches  = partition(bg, n_batch)
        
        #enumerate minibatches
        for i in range(n_batch):

            nll_obj = nLL(pos_batches[i],bg_batches[i])

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

            if (t*sequences_per_batch)%evaluate_after < sequences_per_batch:  #every now and then
                #aucv = auc_evaluate(theta_0, plus_valid, bg_valid)
                #auct = auc_evaluate(theta_0, plus, bg)
                #auc_validation.append(aucv)
                #auc_train.append(auct)
                param_history.append(theta_0)
                #plt_performance(auc_validation, auc_train, param_history, theta_0)
                
                if len(param_history)>5:
                    #if auc_validation[-1] <= auc_validation[-2]:
                    #    reduction += 1

                    #stop when validation set performs worse for at  least red_thr times (variation) and when the parameters are stable
                    variability_index = param_local_fluctuation(param_history)
                    if variability_index<var_thr:
                        np.savetxt(fname='param/'+ plot_name +'.txt', X=np.insert(theta_0,0,f_t))
                        return theta_0, g_t

                    if t>max_iterations:
                        np.savetxt(fname='param/'+ plot_name +'.txt', X=np.insert(theta_0,0,f_t))
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