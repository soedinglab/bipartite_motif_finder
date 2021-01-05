# BMF: Thermodynamic model for de novo bipartite RNA motif discovery

 [![License](https://img.shields.io/github/license/soedinglab/bipartite_motif_finder.svg)](https://choosealicense.com/licenses/gpl-3.0/)
 [![Issues](https://img.shields.io/github/issues/soedinglab/bipartite_motif_finder.svg)](https://github.com/soedinglab/bipartite_motif_finder/issues)

BMF (Bipartite Motif Finder) is an open source tool for finding co-occurences of sequence motifs in genomic sequences. 


BMF is also available as a webserver:

* Link: [bmf.soedinglab.org](https://bmf.soedinglab.org)
* Web server repository: [soedinglab/bmf-webserver](https://github.com/soedinglab/bmf-webserver)



##  Publication

[Sohrabi-Jahromi S and Söding J. Thermodynamic model reveals most RNA-bindingproteins prefer simple and repetitive motifs](https://github.com/soedinglab/bipartite_motif_finder/).



## Installation

### Requirements
  * BMF requires AVX2 extension capable processor. You can check if AVX2 is supported by executing `cat /proc/cpuinfo | grep avx2` on Linux and `sysctl -a | grep machdep.cpu.leaf7_features | grep AVX2` on MacOS).
  * `python>3.6`
  * `numpy`
  * `cython`

### Step-by-step installation:

  1. Create a new conda environment with `python`, `numpy`, and `cython`:
  
    conda create -n bmf python=3.6 numpy cython
    conda activate bmf
       
  2. Install BMF with pip:
  
    pip install https://github.com/soedinglab/bipartite_motif_finder/releases/download/v1.0.0a/bmf_tool-1.0.0.tar.gz

  3. See BMF help page:
  
    bmf --help


## Using BMF

### Motif discovery
    "bmf positives_AAA_CCC.fasta 
    --BGsequences negatives_AAA_CCC.fasta 
    --input_type fasta 
    --output_prefix AAA_CCC
    --motif_length 3 
    --no_tries 1"

### Getting sequence logo
    bmf_logo AAA_CCC --motif_length 3

### Predicting binding to new sequences
    "bmf test_sequences.fasta 
    --input_type fasta 
    --test --model_parameters AAA_CCC 
    --output_prefix predict_test_sequences"
