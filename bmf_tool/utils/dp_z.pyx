cimport cython
cimport numpy as np
ctypedef double c_float_t
import numpy as np
import itertools
from libc.math cimport exp,pow

float_dtype = np.NPY_DOUBLE
np.import_array()

cdef double cpi = np.pi

cpdef generate_kmer_inx(int l, uracil=True):
    '''
    creates an integer index for each kmer
        
    Arguments:
        l -- kmer length (motif core length)          
        uracil -- which alphabet to be used for uracil, set to True if file uses 'U' and False if 'T'

    Returns:
        kmer_inx -- dictionary with kmers as keys and integer representations as values
    '''
        
    if uracil:
        base4 = 'U'
    else:
        base4 = 'T'
        
    cdef dict vals = {'A':0,'C':1,'G':2,base4:3}
    cdef dict kmer_inx = {}
    
    alphabet_length = len(vals.keys())
    
    for p in list(itertools.product(vals.keys(), repeat=l)):
        inx = 0
        for j,base in enumerate(p):
            inx += (alphabet_length**j)*vals[base] 
        kmer_inx[''.join(p)] = inx
    
    return kmer_inx

cpdef generate_struct_inx(int l):
    '''
    creates an integer index for each structural kmer
        
    Arguments:
        l -- kmer length (motif core length)          

    Returns:
        struct_inx -- dictionary with kmers as keys and integer representations as values
    '''
               
    #'.' as single stranded and '(' or ')' as base pairs
    cdef dict vals = {'.':0,'(':1,')':1}
    alphabet_length = 2
    
    cdef dict struct_inx = {}
    
    for p in list(itertools.product(vals.keys(), repeat=l)):
        inx = 0
        for j,base in enumerate(p):
            inx += (alphabet_length**j)*vals[base] 
        struct_inx[''.join(p)] = inx
    
    return struct_inx


cpdef seq2int_cy(str sequence, int l, inx_dict):
    '''
    creates an integer index for each structural kmer
        
    Arguments:
        sequence -- the RNA or structure given as a character string
        l -- kmer length (motif core length)  
        inx_dict -- dictionary with kmers as keys and integer representations as values. 
                    should be compatible with l

    Returns:
        integer_array -- array of integers representing each kmer in the given sequence
    '''
    
    cdef int L = len(sequence)
    integer_array = np.zeros(L, dtype=int)
    
    cdef i
    for i in range(l-1,L):
        kmer = sequence[i-l+1:i+1]
        integer_array[i] = inx_dict[kmer]
    return integer_array        


    

cdef extern from "src_helper_avx_nb.c":
    pass
    
cdef extern from "src_helper_avx_nb.h":
    ctypedef struct DerParams:
        c_float_t* za_Ea_derivatives
        c_float_t* zb_Ea_derivatives
        c_float_t* za_Eb_derivatives
        c_float_t* zb_Eb_derivatives
        c_float_t* za_Eas_derivatives
        c_float_t* zb_Eas_derivatives
        c_float_t* za_Ebs_derivatives
        c_float_t* zb_Ebs_derivatives
        c_float_t* der_a
        c_float_t* der_b
        c_float_t* der_as
        c_float_t* der_bs
        
    void initialize_DerParams(DerParams* params, int L, int no_kmers, int no_struct)
    void deinitialize_DerParams(DerParams* params)

    double sum_array_c(double* arr, int length)
    void sum_mat_rows(double* out, double* mat, int n_row, int n_col)
    
    void assign_za_c(int i, double* za, double* zb, double concentration_times_energy, int l)
    void assign_zb_c(long* x, long* q, int i, double* za, double* zb, double* Eb, double* Ebs, double cab, 
                  double sf, double r, double p, int l)

    void assign_za_E_derivatives_c(long* x, long* q, int i, double* za, double* zb, int l, int no_kmers, int no_struct,
                                 DerParams* params, double* Ea, double* Eas, double cab)
                                 
    void assign_zb_E_derivatives_c(long* x, long* q, int i, double* za, double* zb, int l, int no_kmers, int no_struct,
                                 DerParams* params, double* Eb, double* Ebs, double cab, double sf, double r, double p)


    void assign_za_r_derivative_c(int i, double* za_r_derivatives, double* zb_r_derivatives, double concentration_times_energy, int l)
    void assign_za_p_derivative_c(int i, double* za_p_derivatives, double* zb_p_derivatives, double concentration_times_energy, int l)
    void assign_za_sf_derivative_c(int i, double* za_sf_derivatives, double* zb_sf_derivatives, double concentration_times_energy, int l)

    void assign_zb_r_derivative_c(int i, double* za, double* za_r_derivatives, double* zb_r_derivatives, double energy_b, 
                                         double cab, double sf,double r, double p, int l)
    void assign_zb_p_derivative_c(int i, double* za, double* za_p_derivatives, double* zb_p_derivatives, double energy_b, 
                                         double cab, double sf, double r, double p, int l)
    void assign_zb_sf_derivative_c(int i, double* za, double* za_sf_derivatives, double* zb_sf_derivatives, double energy_b, 
                                         double cab, double sf, double r, double p, int l)

    double cb_c(int, double, double, double)
    double cb_r_derivative_c(int, double, double, double)
    double cb_p_derivative_c(int, double, double, double)
    double cb_sf_derivative_c(int, double, double, double)
    double digamma(double)
    

cdef array_from_pointer(void* ptr, int size):
        cdef np.npy_intp shape_c[1]
        shape_c[0] = <np.npy_intp> size
        ndarray = np.PyArray_SimpleNewFromData(1, shape_c, float_dtype, ptr)
        return ndarray
    
    
@cython.boundscheck(False)  # Deactivate bounds checking   
def DP_Z_cy(double[:] args, long[:] x, long[:] q, int l):
    
    
    cdef int L = len(x)
    
    kmer_inx = generate_kmer_inx(l)
    struct_inx = generate_struct_inx(l)
    

    cdef int no_kmers = len(set(kmer_inx.values()))
    cdef int no_struct = len(set(struct_inx.values()))
    
    cdef double cab = 1.0

    #split args into model parameters in this oder: Ea, Eb, Esa, Esb, sf, r, p
    cdef double[:] Ea = args[0:no_kmers]
    cdef double[:] Eb = args[no_kmers:2*no_kmers]
    cdef double[:] Eas = args[2*no_kmers:2*no_kmers+no_struct]
    cdef double[:] Ebs = args[2*no_kmers+no_struct:2*no_kmers+2*no_struct]
    cdef double sf = args[-3]
    cdef double r = args[-2]
    cdef double p = args[-1]
    
    #initialization of statistical weigths
    cdef double[:] za = np.zeros(L)
    cdef double[:] zb = np.zeros(L)

    cdef int i
    for i in range(0,l):
        zb[i] = 1 
        
        
    #initialization of derivatives
    cdef DerParams params = DerParams()
    initialize_DerParams(&params, L, no_kmers, no_struct)
    

    cdef double[:] za_sf_derivatives = np.zeros(L)
    cdef double[:] zb_sf_derivatives = np.zeros(L)
    
    cdef double[:] za_r_derivatives = np.zeros(L)
    cdef double[:] zb_r_derivatives = np.zeros(L)

    cdef double[:] za_p_derivatives = np.zeros(L)
    cdef double[:] zb_p_derivatives = np.zeros(L)


    cdef int inx
    
    #precompute (binding energy of domain a at position i)*concentration
    cdef double energy_conc_a 
    
    #precompute binding energy of domain b binding at position i
    cdef double energy_b
    
    #dynamic programming calculation of z and derivatives 
    
    
    for i in range(l,L):
        energy_conc_a = cab*exp(-Ea[x[i]]-Eas[q[i]])
        energy_b = exp(-Eb[x[i]]-Ebs[q[i]])
        
        #calculate statistical weights
        
        assign_za_c(i, &za[0], &zb[0], energy_conc_a, l)
        assign_zb_c(&x[0], &q[0], i, &za[0], &zb[0], &Eb[0], &Ebs[0], cab, sf, r, p, l)
        
        #calculate derivatives for all kmers (inx) at each position
        assign_za_E_derivatives_c(&x[0], &q[0], i, &za[0], &zb[0], l, no_kmers, no_struct, 
                                      &params, &Ea[0], &Eas[0], cab)
  
        assign_zb_E_derivatives_c(&x[0], &q[0], i, &za[0], &zb[0], l, no_kmers, no_struct,
                                      &params, &Eb[0], &Ebs[0], cab, sf, r, p)
        
        
        assign_za_sf_derivative_c(i, &za_sf_derivatives[0], &zb_sf_derivatives[0], energy_conc_a, l)
        assign_zb_sf_derivative_c(i, &za[0], &za_sf_derivatives[0], &zb_sf_derivatives[0], energy_b, cab, sf, r, p, l)
        
        assign_za_r_derivative_c(i, &za_r_derivatives[0], &zb_r_derivatives[0], energy_conc_a, l)
        assign_zb_r_derivative_c(i, &za[0], &za_r_derivatives[0], &zb_r_derivatives[0], energy_b, cab, sf, r, p, l)
        
        assign_za_p_derivative_c(i, &za_p_derivatives[0], &zb_p_derivatives[0], energy_conc_a, l)
        assign_zb_p_derivative_c(i, &za[0], &za_p_derivatives[0], &zb_p_derivatives[0], energy_b, cab, sf, r, p, l)

    Z_x = zb[L-1] + sum_array_c(&za[0], L)
    
    #derivative of Z(x)
    #sequence derivatives
    za_Ea_derivatives = array_from_pointer(params.za_Ea_derivatives, no_kmers * L).reshape(L,-1)
    za_Eb_derivatives = array_from_pointer(params.za_Eb_derivatives, no_kmers * L).reshape(L,-1)
    zb_Ea_derivatives = array_from_pointer(params.zb_Ea_derivatives, no_kmers * L).reshape(L,-1)
    zb_Eb_derivatives = array_from_pointer(params.zb_Eb_derivatives, no_kmers * L).reshape(L,-1)
    
    #structure derivatives
    za_Eas_derivatives = array_from_pointer(params.za_Eas_derivatives, no_struct * L).reshape(L,-1)
    za_Ebs_derivatives = array_from_pointer(params.za_Ebs_derivatives, no_struct * L).reshape(L,-1)
    zb_Eas_derivatives = array_from_pointer(params.zb_Eas_derivatives, no_struct * L).reshape(L,-1)
    zb_Ebs_derivatives = array_from_pointer(params.zb_Ebs_derivatives, no_struct * L).reshape(L,-1)


    #calculate derivatives of Ea, Eb, Eas, and Ebs
    d_Ea = zb_Ea_derivatives[L-1,:] + np.sum(za_Ea_derivatives, axis=0)
    d_Eb = zb_Eb_derivatives[L-1,:] + np.sum(za_Eb_derivatives, axis=0)
    d_Eas = zb_Eas_derivatives[L-1,:] + np.sum(za_Eas_derivatives, axis=0)
    d_Ebs = zb_Ebs_derivatives[L-1,:] + np.sum(za_Ebs_derivatives, axis=0)
  
    d_sf = zb_sf_derivatives[L-1] + sum_array_c(&za_sf_derivatives[0], L)
    d_r = zb_r_derivatives[L-1] + sum_array_c(&za_r_derivatives[0], L)
    d_p = zb_p_derivatives[L-1] + sum_array_c(&za_p_derivatives[0], L) 
    
    #concatenate derivatives in this oder: Ea, Eb, Eas, Ebs, sf, r, p
    gradient = np.concatenate([der.ravel() for der in [d_Ea, d_Eb, d_Eas, d_Ebs, np.array([d_sf, d_r, d_p])]])

    deinitialize_DerParams(&params)

    return Z_x, gradient