#!/usr/bin/env python
# coding: utf-8

# ## Learning bi-partite motifs based on a thermodynamic approach
# ### Implements the dynamic programming and the gradient descent

#load functions from the files
exec(compile(open('src/load_libs.py', 'rb').read(), 'src/load_libs.py', 'exec'))
exec(compile(open('src/LL_avx.py', 'rb').read(), 'src/LL_avx.py', 'exec'))
exec(compile(open('src/ADAM_func.py', 'rb').read(), 'src/ADAM_func.py', 'exec'))

#parse input argument(s)
parser = argparse.ArgumentParser()
parser.add_argument('positive_file', type=str)
parser.add_argument('background_file', type=str)
parser.add_argument('no_tries', type=int)
args = parser.parse_args()

positive_file = args.positive_file
background_file = args.background_file
no_tries = args.no_tries



random.seed(42)

background_set = parse_fasta(background_file)
positive_set = parse_fasta(positive_file)

bg_train, bg_valid = partition(background_set, 2)
pos_train, pos_valid = partition(positive_set, 2)



# ### ADAM optimization
for i in range(no_tries):
    np.random.seed(i)
    Ea = np.random.normal(loc=12.0, scale=1.0, size=len(kmer_inx))
    Eb = np.random.normal(loc=12.0, scale=1.0, size=len(kmer_inx))
    sf = np.log(10000)
    r = np.log(np.random.uniform(1,5))
    p = 0

    parameters = np.concatenate([x.ravel() for x in [Ea, Eb, np.array([sf, r, p])]])
    
    var_thr = 0.03
    
    seq_per_batch = 500
    file_name = f'simulation/TAG_ATC_40_08_100percent_{i}'

    maxiter=1000
    x_opt = optimize_adam(pos_train, bg_train, 
                          pos_valid, bg_valid, 
                          var_thr, sequences_per_batch=seq_per_batch, 
                          max_iterations=maxiter, evaluate_after=6000)
