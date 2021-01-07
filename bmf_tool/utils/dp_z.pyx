cimport cython
cimport numpy as np

import numpy as np
import itertools
from libc.math cimport exp,pow


USE_DOUBLE=False

ctypedef float c_float_t  # change double -> float if USE_DOUBLE is set to False

if USE_DOUBLE:
    float_dtype = np.NPY_DOUBLE
    np_float_t = np.float64
else:
    float_dtype = np.NPY_FLOAT32
    np_float_t = np.float32

np.import_array()

cdef c_float_t cpi = np.pi

cpdef generate_kmer_inx(int l):
    cdef dict vals = {'A':0,'C':1,'G':2,'T':3}
    cdef dict kmer_inx = {}
    
    for p in list(itertools.product(vals.keys(), repeat=l)):
        inx = 0
        for j,base in enumerate(p):
            inx += (4**j)*vals[base] 
        kmer_inx[''.join(p)] = inx
    return kmer_inx


cpdef seq2int_cy(str sequence, int l, kmer_inx):
    cdef int L = len(sequence)
    kmer_array = np.zeros(L, dtype=int)
    
    cdef i
    for i in range(l-1,L):
        kmer = sequence[i-l+1:i+1]
        kmer_array[i] = kmer_inx[kmer]
    return kmer_array        


    

cdef extern from "src_helper_avx_nb.c":
    pass
    
cdef extern from "src_helper_avx_nb.h":
    ctypedef struct DerParams:
        c_float_t* za_Ea_derivatives
        c_float_t* zb_Ea_derivatives
        c_float_t* za_Eb_derivatives
        c_float_t* zb_Eb_derivatives
        c_float_t* der_a
        c_float_t* der_b
        
    void initialize_DerParams(DerParams* params, int L, int no_kmers)
    void deinitialize_DerParams(DerParams* params)

    c_float_t sum_array_c(c_float_t* arr, int length)
    void sum_mat_rows(c_float_t* out, c_float_t* mat, int n_row, int n_col)
    
    void assign_za_c(int i, c_float_t* za, c_float_t* zb, c_float_t concentration_times_energy, int l)
    void assign_zb_c(long* x, int i, c_float_t* za, c_float_t* zb, c_float_t* Eb, c_float_t cab, c_float_t sf, c_float_t r, c_float_t p, int l)

    void assign_za_E_derivatives_c(long* x, int i, c_float_t* za, c_float_t* zb, int L, int l, int no_kmers,
                                     DerParams* params, c_float_t* Ea, c_float_t* Eb, c_float_t cab)

    void assign_zb_E_derivatives_c(long* x, int i, c_float_t* za, c_float_t* zb, int L, int l, int no_kmers,
                                     DerParams* params, c_float_t* Ea, c_float_t* Eb, c_float_t cab, c_float_t sf, c_float_t r, c_float_t p)

    void assign_za_r_derivative_c(int i, c_float_t* za_r_derivatives, c_float_t* zb_r_derivatives, c_float_t concentration_times_energy, int l)
    void assign_za_p_derivative_c(int i, c_float_t* za_p_derivatives, c_float_t* zb_p_derivatives, c_float_t concentration_times_energy, int l)
    void assign_za_sf_derivative_c(int i, c_float_t* za_sf_derivatives, c_float_t* zb_sf_derivatives, c_float_t concentration_times_energy, int l)

    void assign_zb_r_derivative_c(int i, c_float_t* za, c_float_t* za_r_derivatives, c_float_t* zb_r_derivatives, c_float_t energy_b, 
                                         c_float_t cab, c_float_t sf,c_float_t r, c_float_t p, int l)
    void assign_zb_p_derivative_c(int i, c_float_t* za, c_float_t* za_p_derivatives, c_float_t* zb_p_derivatives, c_float_t energy_b, 
                                         c_float_t cab, c_float_t sf, c_float_t r, c_float_t p, int l)
    void assign_zb_sf_derivative_c(int i, c_float_t* za, c_float_t* za_sf_derivatives, c_float_t* zb_sf_derivatives, c_float_t energy_b, 
                                         c_float_t cab, c_float_t sf, c_float_t r, c_float_t p, int l)

    c_float_t cb_c(int, c_float_t, c_float_t, c_float_t)
    c_float_t cb_r_derivative_c(int, c_float_t, c_float_t, c_float_t)
    c_float_t cb_p_derivative_c(int, c_float_t, c_float_t, c_float_t)
    c_float_t cb_sf_derivative_c(int, c_float_t, c_float_t, c_float_t)
    c_float_t digamma(c_float_t)
    

cdef array_from_pointer(void* ptr, int size):
        cdef np.npy_intp shape_c[1]
        shape_c[0] = <np.npy_intp> size
        ndarray = np.PyArray_SimpleNewFromData(1, shape_c, float_dtype, ptr)
        return ndarray
    
    
@cython.boundscheck(False)  # Deactivate bounds checking   
def DP_Z_cy(c_float_t[:] args, long[:] x, int l):
    
    cdef int L = len(x)
    
    kmer_inx = generate_kmer_inx(l)

    cdef int no_kmers = len(kmer_inx)
    cdef c_float_t cab = 1.0

    cdef c_float_t[:] Ea = args[0:len(kmer_inx)]
    cdef c_float_t[:] Eb = args[len(kmer_inx):2*len(kmer_inx)]
    cdef c_float_t sf = args[-3]
    cdef c_float_t r = args[-2]
    cdef c_float_t p = args[-1]
    
    #initialization of statistical weigths
    cdef c_float_t[:] za = np.zeros(L, dtype=np_float_t)
    cdef c_float_t[:] zb = np.zeros(L, dtype=np_float_t)

    cdef int i
    for i in range(0,l):
        zb[i] = 1 
        
        

    #initialization of derivatives
    cdef DerParams params = DerParams()
    initialize_DerParams(&params, L, no_kmers)


    cdef c_float_t[:] za_sf_derivatives = np.zeros(L, dtype=np_float_t)
    cdef c_float_t[:] zb_sf_derivatives = np.zeros(L, dtype=np_float_t)
    
    cdef c_float_t[:] za_r_derivatives = np.zeros(L, dtype=np_float_t)
    cdef c_float_t[:] zb_r_derivatives = np.zeros(L, dtype=np_float_t)

    cdef c_float_t[:] za_p_derivatives = np.zeros(L, dtype=np_float_t)
    cdef c_float_t[:] zb_p_derivatives = np.zeros(L, dtype=np_float_t)


    cdef int inx
    
    #precompute (binding energy of domain a binding at position i)*concentration
    cdef c_float_t energy_conc_a 
    
    #precompute binding energy of domain b binding at position i
    cdef c_float_t energy_b
    
    #dynamic programming calculation of z and derivatives 
    
    for i in range(l,L):
        energy_conc_a = cab*exp(-Ea[x[i]])
        energy_b = exp(-Eb[x[i]])
        
        #calculate statistical weights
        assign_za_c(i, &za[0], &zb[0], energy_conc_a, l)
        assign_zb_c(&x[0], i, &za[0], &zb[0], &Eb[0], cab, sf, r, p, l)
        
        #calculate derivatives for all kmers (inx) at each position
        assign_za_E_derivatives_c(&x[0], i, &za[0], &zb[0], L, l, len(kmer_inx), 
                                      &params, &Ea[0], &Eb[0], cab)
        assign_zb_E_derivatives_c(&x[0], i, &za[0], &zb[0], L, l, len(kmer_inx), 
                                      &params, &Ea[0], &Eb[0], cab, sf, r, p)
        
        
        assign_za_sf_derivative_c(i, &za_sf_derivatives[0], &zb_sf_derivatives[0], energy_conc_a, l)
        assign_zb_sf_derivative_c(i, &za[0], &za_sf_derivatives[0], &zb_sf_derivatives[0], energy_b, cab, sf, r, p, l)
        
        assign_za_r_derivative_c(i, &za_r_derivatives[0], &zb_r_derivatives[0], energy_conc_a, l)
        assign_zb_r_derivative_c(i, &za[0], &za_r_derivatives[0], &zb_r_derivatives[0], energy_b, cab, sf, r, p, l)
        
        assign_za_p_derivative_c(i, &za_p_derivatives[0], &zb_p_derivatives[0], energy_conc_a, l)
        assign_zb_p_derivative_c(i, &za[0], &za_p_derivatives[0], &zb_p_derivatives[0], energy_b, cab, sf, r, p, l)

    Z_x = zb[L-1] + sum_array_c(&za[0], L)
    
    #derivative of Z(x)
    
    za_Ea_derivatives = array_from_pointer(params.za_Ea_derivatives, no_kmers * L).reshape(L,-1)
    za_Eb_derivatives = array_from_pointer(params.za_Eb_derivatives, no_kmers * L).reshape(L,-1)
    zb_Ea_derivatives = array_from_pointer(params.zb_Ea_derivatives, no_kmers * L).reshape(L,-1)
    zb_Eb_derivatives = array_from_pointer(params.zb_Eb_derivatives, no_kmers * L).reshape(L,-1)

    d_Ea = zb_Ea_derivatives[L-1,:] + np.sum(za_Ea_derivatives, axis=0)
    d_Eb = zb_Eb_derivatives[L-1,:] + np.sum(za_Eb_derivatives, axis=0)
  
    d_sf = zb_sf_derivatives[L-1] + sum_array_c(&za_sf_derivatives[0], L)
    d_r = zb_r_derivatives[L-1] + sum_array_c(&za_r_derivatives[0], L)
    d_p = zb_p_derivatives[L-1] + sum_array_c(&za_p_derivatives[0], L)       
    gradient = np.concatenate([q.ravel() for q in [d_Ea, d_Eb, np.array([d_sf, d_r, d_p])]])

    deinitialize_DerParams(&params)
    
    return Z_x, gradient