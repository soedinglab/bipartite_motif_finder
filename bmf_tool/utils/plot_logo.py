import argparse
import os
import numpy as np

import matplotlib
matplotlib.use('AGG')

import seaborn as sns
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import nbinom


from .dp_z import generate_kmer_inx



# ## Learning bi-partite motifs based on a thermodynamic approach
# ### Implements the dynamic programming and the gradient descent


def create_parser():
    parser = argparse.ArgumentParser(description='Plots the sequence logo for the bipartite motif found by BMF')
    parser.add_argument('parameter_prefix', type=str, help='path to .param file that specifies model parameters or when multiple parameters exist the common root.')
    parser.add_argument('--motif_length', action="store", type=int, default=3, help='the length of each core in the bipartite motif')
    return parser

#reads the np.array files and extract the parameters
#the distance parameters (last 3) are exp'ed to stay positive
def read_params_for_plot(files):
    n_additional = 1 #non parameters saved at the end (only LL)
    n_exped = 3 #parameters kept positive through log exp trick (sf, r, p)
    params = []
    for f in files:
        param = np.loadtxt(f)
        param[-n_additional-n_exped:-n_additional] = np.exp(param[-n_additional-n_exped:-n_additional])
        param[-n_additional-1] = param[-n_additional-1]/(1+param[-n_additional-1])
        params.append(param)      
    return params

def plot_core_logo(data, ax, color_scheme):
    k=len(data[0][0])
    scalings = [d[1] for d in data]
    total = sum(scalings)
    font_prop = matplotlib.font_manager.FontProperties(family="sans-serif", weight="bold")

    letter_heights = {}
    letter_widths = {}
    for l in "ACGTU":
        lp = matplotlib.textpath.TextPath((0,0), l, size=20, prop=font_prop)
        ext = lp.get_extents()
        letter_heights[l] = ext.y1 - ext.y0
        letter_widths[l] = ext.x1 - ext.x0
    max_letter_height = max(letter_heights.values())
    max_letter_width = max(letter_widths.values())

    letter_scalings = {k:max_letter_height/v for k,v in letter_heights.items()}

    paths = []
    for kmer, scaling in data:
        row = []
        for letter in kmer:
            txt = matplotlib.textpath.TextPath((0,0), letter, size=20, prop=font_prop)
            txt._letter = letter
            row.append(txt)
        paths.append(row)

    scaled_height = total * max_letter_height

    transformed_paths = []
    cur_top = 0
    for row_paths, scaling in zip(paths, scalings):
        cur_x = 0
        for i, path in enumerate(row_paths): 
            sy = scaling/scaled_height * letter_scalings[path._letter]
            sx = 0.03
            path_t = path.transformed(matplotlib.transforms.Affine2D().scale(sx=sx, sy=sy).translate(tx=cur_x, ty=cur_top))
            path_t._letter = path._letter
            transformed_paths.append(path_t)
            ext = path_t.get_extents()
            cur_x += max_letter_width*sx
        cur_top += ext.y1 - ext.y0


    for path in transformed_paths:
        obj = matplotlib.patches.PathPatch(path, edgecolor="black", facecolor=color_scheme[path._letter], lw=0.5)
        ax.add_patch(obj)

    ax.set_ylim(0, 1)
    ax.set_xlim(0, k/2)
    ax.get_xaxis().set_visible(False)
    sns.despine(ax=ax, trim=True)


def energy2prob(energy_series, top_n=5):
    
    probabilities = np.exp(-energy_series)/np.sum(np.exp(-energy_series))
    probabilities_sorted = probabilities.sort_values(ascending=False)
    
    top_kmers = probabilities_sorted[:top_n]
    top_kmers_formatted = [(inx, val) for inx, val in zip(top_kmers.index, top_kmers.values)]
    return top_kmers_formatted

def plot_logo(adam_params, file_name, core_length):

    #assignes each kmer to an index and visa versa
    kmer_inx = generate_kmer_inx(core_length)
    inx_kmer = {y:x for x,y in kmer_inx.items()}
    
    colnames = [inx_kmer[i] for i in range(len(inx_kmer))] + [inx_kmer[i] for i in range(len(inx_kmer))] + ['sf', 'r', 'p'] + ['LL']
    data = pd.DataFrame(adam_params, columns=colnames)
    core1 = data.sort_values(by='LL').iloc[0,:len(kmer_inx)]
    core1_probs = energy2prob(core1, top_n=5)

    core2 = data.sort_values(by='LL').iloc[0,len(kmer_inx):2*len(kmer_inx)]
    core2_probs = energy2prob(core2, top_n=5)

    r = data.sort_values(by='LL')['r'].values[0]
    p = data.sort_values(by='LL')['p'].values[0]

    sns.set_style("ticks")
    sns.despine(trim=True)

    COLOR_SCHEME = {'G': 'orange', 
                    'A': 'red', 
                    'C': 'blue', 
                    'T': 'darkgreen',
                    'U': 'darkgreen'
                   }

    _ , (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(4.5, 1.5))
    plot_core_logo(core1_probs, ax1, color_scheme=COLOR_SCHEME)
    plot_core_logo(core2_probs, ax3, color_scheme=COLOR_SCHEME)

    #plot distance
    mean = ((1-p)*r)/(p)
    xx = np.arange(0,int(mean)+8,1)
    
    _ = ax2.plot(xx, nbinom.pmf(xx, r, p), 'o--',alpha=0.7, color='black')
    _ = ax2.set_xlabel('distance')
    
    #not show y axis in the plot
    ax2.set_frame_on(False)
    _ = ax2.get_yaxis().set_visible(False)
    xmin, xmax = ax2.get_xaxis().get_view_interval()
    ymin, ymax = ax2.get_yaxis().get_view_interval()
    ax2.add_artist(matplotlib.lines.Line2D((xmin, xmax), (ymin, ymin), color='black', linewidth=2))


    _ = ax3.set_yticks(range(0,2))
    _ = ax3.set_yticklabels(np.arange(0,2,1))

    _ = ax3.get_xaxis().set_visible(False)
    _ = ax1.set_ylabel('probability')
    sns.despine(ax=ax2, trim=True)
    sns.despine(ax=ax3, trim=True)

    plt.savefig(file_name + '.pdf', bbox_inches='tight')
    plt.savefig(file_name + '.png', bbox_inches='tight', dpi=150)

def is_bipartite(adam_params, core_length):
    #assignes each kmer to an index and visa versa
    kmer_inx = generate_kmer_inx(core_length)
    inx_kmer = {y:x for x,y in kmer_inx.items()}
    
    colnames = [inx_kmer[i] for i in range(len(inx_kmer))] + [inx_kmer[i] for i in range(len(inx_kmer))] + ['sf', 'r', 'p'] + ['LL']
    data = pd.DataFrame(adam_params, columns=colnames)
    
    r = data.sort_values(by='LL')['r'].values[0]
    p = data.sort_values(by='LL')['p'].values[0]

    prob_zero = nbinom.pmf(0, r, p)

    return prob_zero<0.5


def prefix2params(parameter_prefix, core_length):

    #split the path into directory and filename
    path_to_dir, param_file_names = os.path.split(parameter_prefix)

    #if current directory
    if path_to_dir == '': 
        path_to_dir = '.'
    
    # find all corresponding parameter files
    param_files = [s for s in os.listdir(path_to_dir) if s.startswith(f'{param_file_names}') & s.endswith('txt')]

    #read the parameters
    params = read_params_for_plot([os.path.join(path_to_dir, param_file) for param_file in param_files])
    
    return params


def main():
    parser = create_parser()
    args = parser.parse_args()

    parameter_prefix = args.parameter_prefix
    core_length = args.motif_length

    params = prefix2params(parameter_prefix, core_length)
    plot_logo(params, f'{parameter_prefix}_seqLogo', core_length)
    
    bipartite = is_bipartite(params, core_length)
    if bipartite:
        print('Looks like the motif is bipartite!')
    else:
        print('The motif does not seem to be bipartite! To be sure, you can run BMF with a bigger motif-length.')

    print(f'You can find the BMF logo plot generated at: {parameter_prefix}_seqLogo.pdf')



if __name__ == '__main__':
    main()