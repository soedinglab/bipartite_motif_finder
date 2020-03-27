#!/usr/bin/env python
# coding: utf-8

# ## Learning bi-partite motifs based on a thermodynamic approach
# ### Implements the dynamic programming and the gradient descent

#load functions from the files
exec(compile(open('load_libs.py', 'rb').read(), 'load_libs.py', 'exec'))
exec(compile(open('LL_avx.py', 'rb').read(), 'LL_avx.py', 'exec'))
exec(compile(open('ADAM_func.py', 'rb').read(), 'ADAM_func.py', 'exec'))

#parse input argument(s)
parser = argparse.ArgumentParser()
parser.add_argument('factor')
parser.add_argument('no_tries', type=int)
args = parser.parse_args()

factor = args.factor
no_tries = args.no_tries


#define paths for differnet cycles
celf1_0 = '../RBP_motif_cluster/selex_taipale/CELF1/CELF1_0_TCTAGT40NAAC0_sig.seq_rc.fastq'
celf1_4 = '../RBP_motif_cluster/selex_taipale/CELF1/CELF1_4_construct2_TTCTAC40NCGA_AAG_4_rc.fastq'

if factor.lower()=='celf1':
    bg = celf1_0
    pos = celf1_4
else:
    print('factor not available')
    exit()

    
background_set = parse_fastq(bg)
background_set   = [seq.replace('N', random.sample(['A','T','C','G'],1)[0]) for seq in background_set]

positive_set = parse_fastq(pos)
positive_set = [seq.replace('N', random.sample(['A','T','C','G'],1)[0]) for seq in positive_set]

bg_train, bg_test, bg_valid = partition(background_set, 3)
pos_train, pos_test, pos_valid = partition(positive_set, 3)


# ### ADAM optimization
for i in range(no_tries):
    Ea = np.random.normal(loc=12.0, scale=1.0, size=len(kmer_inx))
    Eb = np.random.normal(loc=12.0, scale=1.0, size=len(kmer_inx))
    sf = np.log(10000)
    D = np.log(np.random.uniform(1,15))
    sig = np.log(np.random.uniform(1,15))

    parameters = np.concatenate([x.ravel() for x in [Ea, Eb, np.array([sf, D, sig])]])

    identifier = random.randint(1,999)
    n=4000
    
    red_thr = 20
    var_thr = 0.03
    
    batch_vector = [500]
    seq_per_batch = batch_vector[i%len(batch_vector)]
    
    plot_name = '%s_newVar_noAUC_%d_ADAM'%(factor, identifier)

    maxiter=1000
    x_opt = optimize_adam(pos_train, bg_train, 
                          random.sample(pos_valid, n), random.sample(bg_valid, n), 
                          red_thr, var_thr, sequences_per_batch=seq_per_batch, 
                          max_iterations=maxiter, evaluate_after=4000)




