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
cycles = np.array([int(s.split('_')[-2]) for s in factor_files])
#note that the first replicate will be chosen in case of multiple choices
last_cycle_file = factor_files[np.argmax(cycles)]
last_cycle_code = last_cycle_file.split('_')[1]
corresponding_bg_file = selex_files[np.array([(last_cycle_code in s)&('ZeroCycle' in s) for s in selex_files])][0]


bg = os.path.join('../rbp_scratch/data', corresponding_bg_file)
pos = os.path.join('../rbp_scratch/data', last_cycle_file)
    
background_set = parse_fastq(bg)
background_set   = [seq.replace('N', random.sample(['A','T','C','G'],1)[0]) for seq in background_set]

positive_set = parse_fastq(pos)
positive_set = [seq.replace('N', random.sample(['A','T','C','G'],1)[0]) for seq in positive_set]

bg_train, bg_test, bg_valid = partition(background_set, 3)
pos_train, pos_test, pos_valid = partition(positive_set, 3)



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
    
    file_name = f'selex/{factor}_4vs0_{i}_cs{core_length}'

    maxiter=1000
    x_opt = optimize_adam(
                            pos_train, bg_train, pos_valid, bg_valid, 
                            core_length=core_length,
                            var_thr=var_thr,
                            sequences_per_batch=seq_per_batch, 
                            max_iterations=maxiter, 
                            evaluate_after=4000
    )




