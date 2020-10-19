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
parser.add_argument('--factor', action="store", type=str)
parser.add_argument('--eclip', action="store", default='', type=str)
parser.add_argument('--outfile', action="store", default='out.pred', type=str)
parser.add_argument('--motif_length', action="store", type=int, default=3)

args = parser.parse_args()

factor = args.factor
eclip=args.eclip
outfile=args.outfile
core_length = args.motif_length

#parse the dataset to find the corresponding files for each factor
if eclip != '':
    print('eclip mode activated')
    fasta_dir = '/cbscratch/salma/eclip_encode/fasta_splitted/'
    fasta_file = os.path.join(fasta_dir, f'{eclip}_split_all.fasta')
else:
    fasta_dir = '/cbscratch/salma/selex_taipale/data_split/'
    fasta_file = os.path.join(fasta_dir, f'{factor}_test_all.fasta')

seq_test = parse_fasta(fasta_file)
seq_test = [seq.replace('N', random.sample(['A','T','C','G'],1)[0]) for seq in seq_test]

#load parameters
if eclip != '':
    param_path = 'param/selex/benchmark_selex_cs{core_length}/'
    print(param_path)
else:
    param_path = 'param/selex/benchmark_selex_train_cs{core_length}/'
    print(param_path)

param_files = [s for s in os.listdir(param_path) if s.startswith(f'{factor}_cs{core_length}')]
params = read_params([os.path.join(param_path, param_file) for param_file in param_files])

#choose best LL solution if multiple runs exist
n_log_likelihoods = [param[-3] for param in params]
tetha = params[np.argmin(n_log_likelihoods)][:-1]

#calculate auroc and ap scores
kmer_inx = generate_kmer_inx(core_length)
y_scores = predict(seq_test, tetha, core_length, kmer_inx)

np.savetxt(fname=outfile, X=y_scores)


