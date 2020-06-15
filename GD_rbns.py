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
parser.add_argument('no_tries', type=int)
args = parser.parse_args()

factor_number = args.factor_number
no_tries = args.no_tries


#collect best concentration information
def index_containing_substring(the_list, substring):
    for i, s in enumerate(the_list):
        if substring in s:
              return i
    return -1

data= pd.read_csv('RBNS_summary.txt', sep='\t')
factors = data.loc[:,'RBP']
concentrations = [int(conc.split(' ')[0]) for conc in data.loc[:,'MostRConc']]
    
    
#define paths for RBNS dataset files
rbp_path = '../rbns_scratch/data'
all_files = [os.listdir(os.path.join(rbp_path, factor)) for factor in factors]
input_files = [[f for f in files if '_input_' in f][0] for files in all_files]

files_flat = [item for sublist in all_files for item in sublist]

if factor_number in range(len(factors)):
    factor = factors[factor_number]
    concentration = concentrations[factor_number]
    print(f'processing {factor}_{concentration}')
    bg = os.path.join(rbp_path,factor,input_files[factor_number])
    pos = os.path.join(rbp_path,factor,files_flat[index_containing_substring(files_flat, f'{factor}_{concentration}')])
else:
    print('factor number exceeds limit')
    exit()

    
random.seed(42)
background_set = parse_fastq(bg)
print(f'background set has {len(background_set)} sequences')
background_set = [seq.replace('N', random.sample(['A','T','C','G'],1)[0]) for seq in background_set]

positive_set = parse_fastq(pos)
print(f'positive set has {len(positive_set)} sequences')
positive_set = [seq.replace('N', random.sample(['A','T','C','G'],1)[0]) for seq in positive_set]

bg_train, bg_valid = partition(background_set, 2)
pos_train, pos_valid = partition(positive_set, 2)


# ### ADAM optimization
for i in range(no_tries):
    np.random.seed(i)
    Ea = np.random.normal(loc=12.0, scale=1.0, size=len(kmer_inx))
    Eb = np.random.normal(loc=12.0, scale=1.0, size=len(kmer_inx))
    sf = np.log(10000)
    D = np.log(np.random.uniform(1,15))
    sig = np.log(np.random.uniform(1,15))

    parameters = np.concatenate([x.ravel() for x in [Ea, Eb, np.array([sf, D, sig])]])
    
    var_thr = 0.03
    
    seq_per_batch = 500
    
    file_name = f'{factor}_{concentration}_batchsize{seq_per_batch}_{i}'

    maxiter=1000
    x_opt = optimize_adam(pos_train, bg_train, 
                          pos_valid, bg_valid, 
                          var_thr, sequences_per_batch=seq_per_batch, 
                          max_iterations=maxiter, evaluate_after=4000)




