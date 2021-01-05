import argparse
import os
import numpy as np

from .utils import (
    parse_sequences, 
    optimize_adam,
    read_params,
    generate_kmer_inx,
    predict,
)



# ## Learning bi-partite motifs based on a thermodynamic approach
# ### Implements the dynamic programming and the gradient descent


def create_parser():
    parser = argparse.ArgumentParser(description='BipartiteMotifFinder can learn two over-represented patterns together with their spacing preference.')
    parser.add_argument('sequences', type=str, help='Train mode: positive sequences enriched with the motif. Test mode: all sequences to be tested')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--BGsequences', type=str, default=None, help='background sequences (only relevant for training).')
    group.add_argument('--test', action="store_true", default=False, help='use this flag to run the testing mode.')
    parser.add_argument('--input_type', action="store", default='fasta', choices=['fasta','fastq','seq'], help='format of input sequences. Can be "fasta", "fastq", and "seq"')
    parser.add_argument('--model_parameters', action="store", type=str, default=None, help='path to .param file that specifies model parameters.')
    parser.add_argument('--motif_length', action="store", type=int, default=3, help='the length of each core in the bipartite motif')
    parser.add_argument('--no_tries', action="store", type=int, default=5, help='the number of times the program is run with random initializations\nNote: only relevant for training')
    parser.add_argument('--output_prefix', action="store", type=str, default=None, help='output file prefix. \nYou can specify a directory e.g. "--output_prefix output_dir/my_prefix"')
    parser.add_argument('--var_thr', action="store", type=float, default=0.03, help='variability threshold to stop ADAM')
    parser.add_argument('--batch_size', action="store", type=int, default=512, help='batch size')
    parser.add_argument('--max_iterations', action="store", type=int, default=1000, help='max number of iterations before stopping ADAM')
    parser.add_argument('--no_cores', action="store", type=int, default=4, help='the numbers of processors to be used in parallel')
    return parser

def main():
    parser = create_parser()
    args = parser.parse_args()

    PSsequences_path = args.sequences
    BGsequences_path = args.BGsequences
    test = args.test
    input_type = args.input_type
    parameters_path = args.model_parameters
    core_length = args.motif_length
    no_tries = args.no_tries
    output_prefix = args.output_prefix
    var_thr = args.var_thr
    batch_size = args.batch_size
    max_iterations = args.max_iterations
    no_cores = args.no_cores

    evaluate_after = 8

    #### REMINDER: TAKE CARE OF U VS T
    use_u = False

    ##############################
    #  train mode                #
    ##############################

    if not test:

        print('BMF training mode')
        #make sure the BG sequences are specified
        if BGsequences_path is None:
            raise ValueError('Please specify background sequences for the train mode using --BGsequences')

        #set the prefix to default if user not defined
        if output_prefix is None:
            output_prefix = 'bipartite'

        #parse sequences
        ps_sequences = parse_sequences(file_name=PSsequences_path, input_type=input_type, use_u=use_u)
        bg_sequences = parse_sequences(file_name=BGsequences_path, input_type=input_type, use_u=use_u)


        print('number of enriched sequences in file: ', len(ps_sequences))
        print('number of background sequences in file: ', len(bg_sequences))
        
        #only keep the number according to max_iterations when too many seqs
        ps_sequences = ps_sequences[:min(batch_size*max_iterations*evaluate_after, len(ps_sequences))]
        bg_sequences = bg_sequences[:min(batch_size*max_iterations*evaluate_after, len(bg_sequences))]

        print('After shortlisting, number of enriched sequences: ', len(ps_sequences))
        print('After shortlisting, number of background sequences: ', len(bg_sequences))

        # ### ADAM optimization
        for i in range(0, no_tries):

            file_prefix = f'{output_prefix}_cs{core_length}_{i}'

            if os.path.isfile(file_prefix+'.txt'):
                print(f'file {file_prefix} already exists. Skipping...',)
                continue 
            
            #initialize with a seed, makes results reproducible
            np.random.seed(i)
            print(f'\nADAM optimization round {i}')
            #random initialization of parameters
            Ea = np.random.normal(loc=12.0, scale=1.0, size=4**core_length)
            Eb = np.random.normal(loc=12.0, scale=1.0, size=4**core_length)
            sf = np.log(10000) #scaling factor is log transformed in the algorithm
            r = np.log(np.random.uniform(1,5))
            p = np.log(np.random.uniform(0,1))


            #### REMOVE
            #r = 1
            #p = 10
            ###########

            parameters = np.concatenate([x.ravel() for x in [Ea, Eb, np.array([sf, r, p])]])
            
            seq_per_batch = batch_size

            x_opt = optimize_adam(
                ps_sequences, bg_sequences,
                parameters=parameters, 
                core_length=core_length,
                var_thr=var_thr,
                sequences_per_batch=seq_per_batch, 
                max_iterations=max_iterations, 
                evaluate_after=evaluate_after,
                no_cores=no_cores,
                save_files=True,
                file_name=file_prefix
            )


    ##############################
    #  test mode                 #
    ##############################
    else:
        #make sure the model parameters are specified
        if parameters_path is None:
            raise ValueError('Please specify model parameters for the test mode using --model_parameters')

        #set the prefix to default if user not defined
        if output_prefix is None:
            output_prefix = 'bipartite'

        sequences = parse_sequences(file_name=PSsequences_path, input_type=input_type, use_u=use_u)

        #split the path into directory and filename
        path_to_dir, param_file_names = os.path.split(parameters_path)
        
        # find all corresponding parameter files
        param_files = [s for s in os.listdir(path_to_dir) if s.startswith(f'{param_file_names}') & s.endswith('txt')]

        #read the parameters
        params = read_params([os.path.join(path_to_dir, param_file) for param_file in param_files])

        #choose best LL solution if multiple runs exist
        n_log_likelihoods = [param[-3] for param in params]
        tetha = params[np.argmin(n_log_likelihoods)][:-1]

        #calculate auroc and ap scores
        kmer_inx = generate_kmer_inx(core_length)
        y_scores = predict(sequences, tetha, core_length, kmer_inx)

        np.savetxt(fname=f'{output_prefix}.predictions', X=y_scores)


if __name__ == '__main__':
    main()