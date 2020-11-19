def auc_evaluate_arbitrary_length(param, ps_valid, bg_valid, core_length, kmer_inx, length=40):
    
    ps = split_seqs(ps_valid, length)
    bg = split_seqs(bg_valid, length)
    
    z_ps = np.zeros(len(ps))
    z_bg = np.zeros(len(bg))
    
    seq_ps = [[seq2int_cy('A' + x, core_length, kmer_inx) for x in arr] for arr in ps]
    seq_bg = [[seq2int_cy('A' + x, core_length, kmer_inx) for x in arr] for arr in bg]
    
    n_pos = 3
    #exp parameters to make sure they are positive
    args = param.copy()
    args[-n_pos:-1] = np.exp(args[-n_pos:-1])
    exp_p = np.exp(args[-1])
    args[-1] = exp_p/(1+exp_p)
        
    for i, arr in enumerate(seq_ps):
        z_arr = []
        for x in arr:
            z, _ = DP_Z_cy(args, x, core_length)
            z_arr.append(z)
        z_ps[i] = np.max(z_arr)
    
    for i, arr in enumerate(seq_bg):
        z_arr = []
        for x in arr:
            z, _ = DP_Z_cy(args, x, core_length)
            z_arr.append(z)
        z_bg[i] = np.max(z_arr)
        
    y_true = np.append(np.ones(len(ps)), np.zeros(len(bg)))
    y_score = np.append(z_ps, z_bg)
    
    fpr_grd, tpr_grd, _ = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)
    
    return fpr_grd, tpr_grd, auc


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

def predict(test_sequences, tetha, core_length, kmer_inx):
    z = np.zeros(len(test_sequences))
    
    seq_int = [seq2int_cy('A' + x, core_length, kmer_inx) for x in test_sequences]
    
    n_pos = 3
    #exp parameters to make sure they are positive
    args = tetha.copy()
    args[-n_pos:-1] = np.exp(args[-n_pos:-1])
    exp_p = np.exp(args[-1])
    args[-1] = exp_p/(1+exp_p)
        
    for i, x in enumerate(seq_int):
        z[i], _ = DP_Z_cy(args, x, core_length)
        
    return z

def evaluate_test_set(param, plus, bg, core_length, kmer_inx):
    z_ps = np.zeros(len(plus))
    z_bg = np.zeros(len(bg))
    
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


def plt_performance(plus, bg, param_history, core_length, kmer_inx, file_name, evaluate_after):
    
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
        
    # AUC plots ================================
    
    fpr_t, tpr_t, auct = auc_evaluate(theta, plus, bg, core_length, kmer_inx)
        
    ax1.plot([0, 1], [0, 1], 'k--')
    ax1.plot(fpr_t, tpr_t, label=f'training (AUC={auct:.2f})')
    ax1.set_xlabel('False positive rate')
    ax1.set_ylabel('True positive rate')
    ax1.set_title('ROC curve')
    ax1.legend(loc='best')

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
        
        ax2.plot(xx, core1_hist, color='#097d87', label='core1 %s E'%kmer1, ls='--')
        ax2.plot(xx, core2_hist, color='#873f0b', label='core2 %s E'%kmer2, ls='--')
    
    ax2.set_xlabel('n\'th iteration')
    ax2.legend()
        
    # plot r, p, and SF ========================
    
    sf_hist = [np.exp(arr[-3]) for arr in param_history]
    r_hist = [np.exp(arr[-2]) for arr in param_history]
    p_hist = [1/(1+np.exp(-arr[-1])) for arr in param_history]
    
    mean_hist = [((1-p)*r)/(p) for r,p in zip(r_hist, p_hist)]
    mode_hist = [max(0,int(((1-p)*(r-1))/p)) for r,p in zip(r_hist, p_hist)]

    ax3.set_xlabel('n\'th iteration')
    ax3.plot(xx, mean_hist, color='#092e87', label='mean of NB distribution')  
    ax3.plot(xx, mode_hist, color='#870b1b', label='mode of NB distribution')
    
    ax3.legend() 
    
    #================================================
    
    plt.savefig(f'{file_name}.pdf', bbox_inches='tight')
    return auct
    


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



def optimize_adam(plus, bg, core_length=3, var_thr=0.05, 
                  sequences_per_batch=100, max_iterations=1000, evaluate_after=None, save_files=True, file_name='bipartite'):

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
            
            #every now and then: evaluate after $evaluate_after iterations
            if t%evaluate_after == 0:                
                #track parameter evolution
                param_history.append(theta_0)
                
                #minimum iterations before checking for convergence: min_iter                
                if len(param_history)>min_iter:

                    #stop when the parameters are stable (see param_local_fluctuation)
                    #OR stop when it took too long ($max_iterations)
                    
                    variability_index = param_local_fluctuation(param_history)
                    print('variability_index', variability_index)
                    if variability_index<var_thr or t>max_iterations:
                        if save_files:
                            _ = plt_performance(plus, bg, param_history, core_length, kmer_inx, file_name, evaluate_after)
                            np.savetxt(fname=file_name +'.txt', X=np.append(theta_0,[f_t]))
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

def parse_sequences(file_name, input_type, use_u=False):

    # calling the right parser according to datatype
    if input_type == 'fastq':
        sequences = parse_fastq(file_name)
    elif input_type == 'fasta':
        sequences = parse_fasta(file_name)
    elif input_type == 'seq':
        sequences = parse_seq(file_name)
    else:
        raise ValueError('input_type is not valid')

    #base2 depends on if U or T is used
    if use_u:
        base2 = 'U'
    else:
        base2 = 'T'
    
    #replace N with random nucleotides
    sequences = [seq.replace('N', random.sample(['A',base2,'C','G'],1)[0]) for seq in sequences]
    return sequences

    

def parse_clip_and_split(file_name, length=40):
    input_seq_iterator = SeqIO.parse(file_name, "fasta")
    sequences = [str(record.seq) for record in input_seq_iterator]
    sequences_split = [[s[i:i+length] for i in np.arange(0,len(s)-length+1,5)] for s in sequences]
    
    return [item.upper().replace('U','T') for sublist in sequences_split for item in sublist]

def find_start_index_clip(seq):
    for i , c in enumerate(seq):
        if c.isupper():
            return i
    return 0

def parse_clip(file_name):
    input_seq_iterator = SeqIO.parse(file_name, "fasta")
    sequences = [str(record.seq) for record in input_seq_iterator]  
    
    clipped_sequences = []
    for seq in sequences:
        start_inx = find_start_index_clip(seq)
        if start_inx + 60 < len(seq):
            end_inx = start_inx + 60
        else:
            end_inx = len(seq)
            start_inx = end_inx - 60  
        clipped_sequences.append(seq[start_inx:end_inx].upper().replace('U','T'))
    return clipped_sequences

def split_seqs(sequences, length=40):
    return [[s[i:i+length] for i in np.arange(0,len(s)-length+1,5)] for s in sequences]

def read_params(files):
    params = []
    for f in files:
        param = np.loadtxt(f)
        params.append(param)      
    return params