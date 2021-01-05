
from Bio import SeqIO
import random

# Sequence Parsers
def parse_fastq(file_name):
    input_seq_iterator = SeqIO.parse(file_name, "fastq")
    return [str(record.seq) for record in input_seq_iterator]

def parse_fasta(file_name):
    input_seq_iterator = SeqIO.parse(file_name, "fasta")
    return [str(record.seq) for record in input_seq_iterator]

def parse_seq(file_name):
    with open(file_name,'r') as f:
        seq = [line.rstrip() for line in f]
    return seq

#general sequence parser
def parse_sequences(file_name, input_type, use_u=False):

    # calling the right parser according to datatype
    if input_type == 'fastq':
        sequences = parse_fastq(file_name)
    elif input_type == 'fasta':
        sequences = parse_fasta(file_name)
    elif input_type == 'seq':
        sequences = parse_seq(file_name)
    else:
        raise ValueError('input_type is not valid')

    #base2 depends on if U or T is used
    if use_u:
        base2 = 'U'
    else:
        base2 = 'T'
    
    #replace N with random nucleotides
    sequences = [seq.replace('N', random.sample(['A',base2,'C','G'],1)[0]) for seq in sequences]
    return sequences

#checks validity of first n sequences
def check_sequences_top_n(filename, file_format, first_n=10):

    # calling the right parser according to datatype
    if file_format == 'fastq':
        sequences = parse_fastq(filename)
        
    elif file_format == 'fasta':
        sequences = parse_fasta(filename)
        
    elif file_format == 'seq':
        sequences = parse_seq(filename)
    else:
        raise ValueError('input_type is not valid')
        
    sequences = sequences[:first_n]
    
    #check if sequences have same length
    lengths = [len(seq) for seq in sequences]
    
    if min(lengths) != max(lengths):
        raise ValueError('Input sequences have different lengths.')
        
    #check if nucleotides are in alphabet
    legal_bases = ['A', 'C', 'G', 'T', 'U', 'N']
    concatenated_reads = ''.join(sequences).upper()
    reads_alphabet = set(concatenated_reads)
    for character in reads_alphabet:
        if character not in legal_bases:
            raise ValueError(f'Input sequences have non-conventional base: {character}')
            
    return True


#checks file format validity
def is_format(filename, file_format):
    with open(filename, "r") as handle:
        sequences = SeqIO.parse(handle, file_format)
        return any(sequences)  # False when sequences is empty, i.e. wasn't a FASTA file


#file validator (checks format and sequence validity)
def file_validator(filename, file_format, first_n=10):
    if file_format in ['fasta', 'fastq']:
        if not is_format(filename, file_format):
            raise ValueError('File format is incorrect.')
            
    return check_sequences_top_n(filename, file_format, first_n)