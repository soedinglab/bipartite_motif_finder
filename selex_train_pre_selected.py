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
parser.add_argument('factor_number', type=int)
parser.add_argument('--motif_length', action="store", type=int, default=3)
parser.add_argument('--no_tries', action="store", type=int, default=5)
parser.add_argument('--all_sequences', default=False, action='store_true')

args = parser.parse_args()

factor_number = args.factor_number
core_length = args.motif_length
no_tries = args.no_tries
all_sequences = args.all_sequences

if all_sequences:
    print('building a bipartite model on all sequences')
else:
    print('building a bipartite model on training set')


#parse the dataset to find the corresponding files for each factor
fasta_dir = '/cbscratch/salma/selex_taipale/data_fasta/'
file_table = pd.read_csv('/cbscratch/salma/selex_taipale/ps_and_bg_file_table.txt', sep='\t', header=None) #name of all files
factor = file_table.iloc[factor_number,0]

if all_sequences:
    bg_file = os.path.join(fasta_dir, file_table.iloc[factor_number,1])
    ps_file = os.path.join(fasta_dir, file_table.iloc[factor_number,2])
else:
    bg_file = f'/cbscratch/salma/selex_taipale/data_split/{factor}_train_bg.fasta'
    ps_file = f'/cbscratch/salma/selex_taipale/data_split/{factor}_train_ps.fasta'





background_set = parse_fasta(bg_file)
background_set   = [seq.replace('N', random.sample(['A','T','C','G'],1)[0]) for seq in background_set]

positive_set = parse_fasta(ps_file)
positive_set = [seq.replace('N', random.sample(['A','T','C','G'],1)[0]) for seq in positive_set]

# ### ADAM optimization
for i in range(0, no_tries):
    
    np.random.seed(i)
    
    Ea = np.random.normal(loc=12.0, scale=1.0, size=4**core_length)
    Eb = np.random.normal(loc=12.0, scale=1.0, size=4**core_length)
    sf = np.log(10000)
    r = np.log(np.random.uniform(1,5))
    p = np.log(np.random.uniform(0,1))

    parameters = np.concatenate([x.ravel() for x in [Ea, Eb, np.array([sf, r, p])]])
    
    var_thr = 0.03
    
    seq_per_batch = 500
    
    if all_sequences:
        file_name = f'selex/benchmark_selex_cs{core_length}/{factor}_cs{core_length}_{i}'
    else:
        file_name = f'selex/benchmark_selex_train_cs{core_length}/{factor}_cs{core_length}_{i}'

    maxiter=1000
    x_opt = optimize_adam(
                            positive_set, background_set, 
                            [], [], 
                            core_length=core_length,
                            var_thr=var_thr,
                            sequences_per_batch=seq_per_batch, 
                            max_iterations=maxiter, 
                            evaluate_after=4000,
                            save_files=True
    )




