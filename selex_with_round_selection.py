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

args = parser.parse_args()

factor_number = args.factor_number
core_length = args.motif_length
no_tries = args.no_tries


#parse the dataset to find the corresponding files for each factor
selex_files = np.loadtxt('selex_files.txt', dtype=str) #name of all files

#first part of file_name is the protein name
factors = np.unique([s.split('_')[0] for s in selex_files]) 

#select the factor based on the input
factor = factors[factor_number]

#files that correspond to this factor
factor_files = selex_files[np.array([factor in s for s in selex_files])]
factor_barcodes = np.unique([s.split('_')[1] for s in factor_files])
files_by_bc = []
for bc in factor_barcodes:
    files = factor_files[np.array([bc in s for s in factor_files])]
    corresponding_bg_file = selex_files[np.array([(bc in s)&('ZeroCycle' in s) for s in selex_files])][0]
    files_with_bg = np.append(files,corresponding_bg_file)
    files_with_bg_sorted_by_cycle = files_with_bg[np.argsort([int(f.split('_')[3]) for f in files_with_bg])]
    files_by_bc.append(files_with_bg_sorted_by_cycle)
    
    
selex_files = files_by_bc[0]
cycles = np.array([int(s.split('_')[3]) for s in selex_files])
avg_auc = []

selex_files = [os.path.join('/cbscratch/salma/selex_taipale/data_struct', f.replace('fastq','struct')) for f in selex_files]

'''
train_sets = []
test_sets = []

#separate trainig and test sets
for i in range(len(selex_files)):
    
    sequences = parse_fastq(selex_files[i])
    sequences = [seq.replace('N', random.sample(['A','T','C','G'],1)[0]) for seq in sequences]
    
    #calculate the size for x fraction of dataset
    fraction = 0.05
    no_seqs  = int(len(sequences)*fraction)
    
    #partition the test and training sets
    train, test = partition(sequences[:no_seqs], 2)
    
    train_sets.append(train)
    test_sets.append(test)
    
    
#parameter initialization
np.random.seed(0)

Ea = np.random.normal(loc=12.0, scale=1.0, size=4**core_length)
Eb = np.random.normal(loc=12.0, scale=1.0, size=4**core_length)
sf = np.log(10000)
r = np.log(np.random.uniform(1,15))
p = 0

parameters = np.concatenate([x.ravel() for x in [Ea, Eb, np.array([sf, r, p])]])

var_thr = 0.03
seq_per_batch = 500

#calculate auc values for consecutive  pairs 
for bg_inx in range(0,len(selex_files)-1):

    maxiter=300
    
    file_name = f'dev_{bg_inx}_{maxiter}iterations_{core_length}cl'
    theta_0, g_t  = optimize_adam(
                                        train_sets[bg_inx+1], train_sets[bg_inx], 
                                        test_sets[bg_inx+1], test_sets[bg_inx], 
                                        core_length=core_length,
                                        var_thr=var_thr, 
                                        sequences_per_batch=seq_per_batch, 
                                        max_iterations=maxiter, 
                                        evaluate_after=4000,
                                        save_files=False
                                        )
    
    kmer_inx = generate_kmer_inx(core_length)
    auc_list = []
    
    #calculate the performance (AUC) of the obtained parameters on all consecutive selex rounds
    for bg_inx in range(0,len(selex_files)-1):
        _,_,auc = auc_evaluate(theta_0, test_sets[bg_inx+1], test_sets[bg_inx], core_length, kmer_inx)
        auc_list.append(auc)

    #calculate average auc obtained with theta_0
    avg_auc.append(np.mean(auc_list))
    

np.savetxt(fname=f'param/selex/cl{core_length}'+ factor +'_avg_auc.txt', X=np.array(avg_auc))
'''
try:
    avg_auc = np.loadtxt(f'param/selex/cl{core_length}'+ factor +'_avg_auc.txt')

    bg_round = np.argmax(avg_auc)
    pos_round = bg_round+1
except:
    pos_round = -1
    bg_round = -2

X_ps = parse_rnafold(selex_files[pos_round])
X_bg = parse_rnafold(selex_files[bg_round])

y_ps = np.ones(len(X_ps))
y_bg = np.ones(len(X_bg))

ps_train, ps_test, _ , _ = train_test_split(X_ps, y_ps, test_size=0.2, random_state=42)
bg_train, bg_test, _ , _ = train_test_split(X_bg, y_bg, test_size=0.2, random_state=42)

# ### ADAM optimization
for i in range(0, no_tries):
    
    np.random.seed(i)
    
    core_length = 3
    kmer_inx = generate_kmer_inx(core_length)
    struct_inx = generate_struct_inx(core_length)
    no_kmers = len(set(kmer_inx.values()))
    no_struct = len(set(struct_inx.values()))


    Ea = np.random.normal(loc=12.0, scale=1.0, size=no_kmers)
    Eb = np.random.normal(loc=12.0, scale=1.0, size=no_kmers)
    Eas = np.random.normal(loc=12.0, scale=1.0, size=no_struct)
    Ebs = np.random.normal(loc=12.0, scale=1.0, size=no_struct)

    sf = np.log(10000)
    r = np.log(np.random.uniform(1,5))
    p = np.log(np.random.uniform(0,1))

    parameters = np.concatenate([x.ravel() for x in [Ea, Eb, Eas, Ebs, np.array([sf, r, p])]])
    
    var_thr = 0.03
    
    seq_per_batch = 512
    
    file_name = f'selex/cl{core_length}_struct/{factor}_{cycles[pos_round]}vs{cycles[bg_round]}_cs{core_length}_{i}'

    maxiter=1000
    x_opt = optimize_adam(
                            ps_train, bg_train, 
                            ps_test, bg_test, 
                            core_length=core_length,
                            var_thr=var_thr,
                            sequences_per_batch=seq_per_batch, 
                            max_iterations=maxiter, 
                            evaluate_after=4000,
                            save_files=True
    )



