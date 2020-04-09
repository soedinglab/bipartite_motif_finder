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
parser.add_argument('factor_number', type=int)
parser.add_argument('no_tries', type=int)
args = parser.parse_args()

factor_number = args.factor_number
no_tries = args.no_tries


#define paths for RBNS dataset files
rbp_path = '../rbns_scratch/data'
factors = np.sort(os.listdir(rbp_path)) #list available factors

all_files = [os.listdir(os.path.join(rbp_path, factor)) for factor in factors]

pulldown_files = [np.sort(files)[2] for files in all_files]  #take 320 concentration when available otherwise another concentration
input_files = [[f for f in files if '_input_' in f][0] for files in all_files]
concentrations = [f.split('_')[1] for f in pulldown_files] #see which concentration we took


if factor_number in range(len(factors)):
    factor = factors[factor_number]
    concentration = concentrations[factor_number]
    print(f'processing {factor}')
    bg = os.path.join(rbp_path,factor,input_files[factor_number])
    pos = os.path.join(rbp_path,factor,pulldown_files[factor_number])
else:
    print('factor number exceeds limit')
    exit()

    
background_set = parse_fastq(bg)
background_set   = [seq.replace('N', random.sample(['A','T','C','G'],1)[0]) for seq in background_set]

positive_set = parse_fastq(pos)
positive_set = [seq.replace('N', random.sample(['A','T','C','G'],1)[0]) for seq in positive_set]

bg_train, bg_valid = partition(background_set, 2)
pos_train, pos_valid = partition(positive_set, 2)


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
    
    seq_per_batch = 500
    
    plot_name = '%s_%s_%d_ADAM'%(factor, concentration, identifier)

    maxiter=1000
    x_opt = optimize_adam(pos_train, bg_train, 
                          random.sample(pos_valid, n), random.sample(bg_valid, n), 
                          red_thr, var_thr, sequences_per_batch=seq_per_batch, 
                          max_iterations=maxiter, evaluate_after=4000)




