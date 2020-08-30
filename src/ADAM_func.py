def auc_evaluate(param, plus, bg, core_length, kmer_inx, struct_inx):
    z_plus = np.zeros(len(plus))
    z_bg = np.zeros(len(bg))
    
    x_ps = [seq2int_cy('A' + x, core_length, kmer_inx) for x in plus[:,0]]
    x_bg = [seq2int_cy('A' + x, core_length, kmer_inx) for x in bg[:,0]]
    q_ps = [seq2int_cy('.' + x, core_length, struct_inx) for x in plus[:,1]]
    q_bg = [seq2int_cy('.' + x, core_length, struct_inx) for x in bg[:,1]]
    
    n_pos = 3
    #exp parameters to make sure they are positive
    args = param.copy()
    args[-n_pos:-1] = np.exp(args[-n_pos:-1])
    exp_p = np.exp(args[-1])
    args[-1] = exp_p/(1+exp_p)
        
    for i in range(len(x_ps)):
        z_plus[i], _ = DP_Z_cy(args, x_ps[i], q_ps[i], core_length)
    
    for i in range(len(x_bg)):
        z_bg[i], _ = DP_Z_cy(args, x_bg[i], q_bg[i], core_length)
        
    y_true = np.append(np.ones(len(plus)), np.zeros(len(bg)))
    y_score = np.append(z_plus, z_bg)
    
    fpr_grd, tpr_grd, _ = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)
    
    return fpr_grd, tpr_grd, auc



def partition(total_no, sequences_per_batch):
    batch_coords = []
    t = 0
    while t*sequences_per_batch<total_no:
        batch_coords.append((t*sequences_per_batch, min((t+1)*sequences_per_batch,total_no))) 
        t += 1
    no_batches = len(batch_coords)
    return no_batches, batch_coords


def plt_performance(plus, bg, plus_valid, bg_valid, param_history, core_length, kmer_inx, struct_inx):
    
    inx_kmer = dict((v,k) for k,v in kmer_inx.items())
    no_kmers = len(set(kmer_inx.values()))
    
    inx_struct = dict((v,k) for k,v in struct_inx.items())
    no_structs = len(set(struct_inx.values()))
        
    fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(12,3.5))
    
    theta = param_history[-1]
    x = np.arange(1, len(param_history)+1, 1)

    # get current motif =======================
    
    core1_s = {}
    core2_s = {}    
    for i in range(no_kmers):
        core1_s[inx_kmer[i]] = theta[i]
        core2_s[inx_kmer[i]] = theta[i+no_kmers]

    core1_s = pd.Series(core1_s).sort_values(ascending=True)  
    core2_s = pd.Series(core2_s).sort_values(ascending=True)
    
    core1_q = {}
    core2_q = {}    
    for i in range(no_structs):
        core1_q[inx_struct[i]] = theta[i]
        core2_q[inx_struct[i]] = theta[i+no_structs]

    core1_q = pd.Series(core1_q).sort_values(ascending=True)  
    core2_q = pd.Series(core2_q).sort_values(ascending=True)
    
    r = np.exp(theta[-2])
    p = 1/(1+np.exp(-theta[-1]))
    

    # set plot title ===========================

    local_fluctuation = param_local_fluctuation(param_history, no_kmers)
    fig.suptitle(f'{core1_s.index[0]}({core1_s.values[0]:.2f}) -- r={r:.1f},p={p:.1f} -- {core2_s.index[0]}({core2_s.values[0]:.2f})\n' +
    f'{core1_s.index[1]}({core1_s.values[1]:.2f}) ----------------- {core2_s.index[1]}({core2_s.values[1]:.2f})\n' +
    f'{core1_s.index[2]}({core1_s.values[2]:.2f}) ----------------- {core2_s.index[2]}({core2_s.values[2]:.2f})\n' +
    f'variation index: {local_fluctuation:.2f}', horizontalalignment='left', fontsize=12, x=0.1, y=1.2)
        
    # AUC plots ================================
    
    fpr_v, tpr_v, aucv = auc_evaluate(theta, plus_valid, bg_valid, core_length, kmer_inx, struct_inx)
    fpr_t, tpr_t, auct = auc_evaluate(theta, plus, bg, core_length, kmer_inx, struct_inx)
        
    ax1.plot([0, 1], [0, 1], 'k--')
    ax1.plot(fpr_v, tpr_v, label=f'validation (AUC={aucv:.2f})')
    ax1.plot(fpr_t, tpr_t, label=f'training (AUC={auct:.2f})')
    ax1.set_xlabel('False positive rate')
    ax1.set_ylabel('True positive rate')
    ax1.set_title('ROC curve')
    ax1.legend(loc='best')

    # Plot binding energies of best bound k-mer per core ====
    
    kmer1 = core1_s.index[0]
    kmer2 = core2_s.index[0]
    
    core1_hist = [arr[kmer_inx[kmer1]] for arr in param_history]
    core2_hist = [arr[kmer_inx[kmer2]+ no_kmers] for arr in param_history]
    
    ax2.plot(x, core1_hist, color='#3761a3', label='c1 seq: %s'%kmer1)
    ax2.plot(x, core2_hist, color='#3761a3', label='c2 seq: %s'%kmer2, linestyle='--')
    
    struct1 = core1_q.index[0]
    struct2 = core2_q.index[0]
    
    core1_hist = [arr[struct_inx[struct1]+ no_kmers*2] for arr in param_history]
    core2_hist = [arr[struct_inx[struct2]+ no_kmers*2 + no_structs] for arr in param_history]
    
    ax2.plot(x, core1_hist, color='#9e2f3c', label='c1 struct %s'%struct1.replace(')','|'))
    ax2.plot(x, core2_hist, color='#9e2f3c', label='c2 struct %s'%struct2.replace(')','|'), linestyle='--')
    
    ax2.set_xlabel('n\'th iteration')
    ax2.legend()
        
    # plot r, p, and SF ========================
    
    sf_hist = [np.exp(arr[-3]) for arr in param_history]
    r_hist = [np.exp(arr[-2]) for arr in param_history]
    p_hist = [1/(1+np.exp(-arr[-1])) for arr in param_history]
    
    ax3.set_xlabel('n\'th iteration')
    ax3.plot(x, r_hist, color='blue', label='r')  
    #ax3.plot(x, p_hist, color='red', label='p')
    
    ax32 = ax3.twinx()  # instantiate a second axes that shares the same x-axis
    ax32.set_ylabel('p', color='red')  # we already handled the x-label with ax1
    ax32.plot(x, p_hist, color='red', label='p')
    ax32.tick_params(axis='y', labelcolor='red')    
    
    ax3.legend() 
    ax32.legend()
    
    #================================================
    
    plt.savefig('plots/'+ file_name +'.pdf', bbox_inches='tight')
    return auct, aucv
    


#returns a fluctuation score [0-1] max representing parameter variation in the last 5 iterations.

def param_local_fluctuation(param_history, no_kmers):
    
    last_params = list(param_history[-1])
    
    #index of best bound sequence kmer
    min_energy_inx = last_params.index(min(last_params[:no_kmers*2]))
    
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
                  sequences_per_batch=256, max_iterations=1000, evaluate_after=None, save_files=True):
    
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

    f_t = 42 #initialize with random number
    epoch = 0
    
    min_iter = 20 #minimum rounds of parameter recording before checking for convergence

    #auc array tracks auc values
    param_history = []    
    
    #calculate the stretch of reduced performance
    reduction = 0
    
    #if None, assign an epoch evaluation scheme
    if evaluate_after is None:
        evaluate_after = len(plus)
    
    # create kmer-index 
    kmer_inx = generate_kmer_inx(core_length)
    struct_inx = generate_struct_inx(core_length)
    
    no_kmers = len(set(kmer_inx.values())) 
    
    #split into minibatches
    #create index lists for each minibatch for positive examples
    no_batches_ps, batch_coords_ps = partition(plus.shape[0], sequences_per_batch)

    #create index lists for each minibatch for background examples 
    #(ensures each minibatch has similar number of pos and bg)
    no_batches_bg, batch_coords_bg = partition(bg.shape[0], sequences_per_batch)

    t = 0

    #enumerate minibatches
    while True:

        t += 1
        
        #calculate indices for each batch of ps and bg sequences 
        inx_start_ps, inx_end_ps = batch_coords_ps[t%no_batches_ps]
        inx_start_bg, inx_end_bg = batch_coords_bg[t%no_batches_bg]
        
        #create nLL object with the current minibatch
        nll_obj = nLL(plus[inx_start_ps:inx_end_ps], bg[inx_start_bg:inx_end_bg], core_length)

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

                variability_index = param_local_fluctuation(param_history, no_kmers)
                
                if variability_index<var_thr or t>max_iterations:
                    if save_files:
                        auct, aucv = plt_performance(plus, bg, plus_valid, bg_valid, param_history, core_length, kmer_inx, struct_inx)
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

def parse_rnafold(input_path, uracil=True):
    '''
    parses RNAfold single-file output which looks like this:
    >header
    GGGAGGAUUGCGUUAGUUAUGUUAUGAGUUACGUUGUGGG
    ((.((((.(.((((...)))).).)))).))... ( -3.80)
    
    input:
        input_path: path to RNAfold output file
        
    output:
        seq_struct_tuple: tuple of sequence-structure pairs        
    '''
    
    #read all lines
    with open(input_path,'r') as f:
        seq = [line.rstrip() for line in f]
        
    #remove the energy from the end of structure output
    if uracil:
        base4 = 'U'
    else:
        base4 = 'T'
        
    sequences = [sequence.replace('N', random.sample(['A','C','G', base4],1)[0]) for sequence in seq[1::3]]
    structures = [structure.split(' ')[0] for structure in seq[2::3]]
    
    
    seq_struct_tuple = np.array(list(zip(sequences, structures)))
    
    return seq_struct_tuple