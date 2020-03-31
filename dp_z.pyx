cimport cython
cimport numpy as np
ctypedef double c_float_t
import numpy as np
import itertools
from libc.math cimport exp,pow

float_dtype = np.NPY_DOUBLE
np.import_array()

cdef double cpi = np.pi
cdef int l = 3 #l_A=l_B=3 nucleotides

cpdef generate_kmer_inx():
    cdef dict vals = {'A':0,'C':1,'G':2,'T':3}
    cdef dict kmer_inx = {}
    
    for p in list(itertools.product(vals.keys(), repeat=l)):
        inx = 0
        for j,base in enumerate(p):
            inx += (4**j)*vals[base] 
        kmer_inx[''.join(p)] = inx
    return kmer_inx

kmer_inx = generate_kmer_inx()
inx_kmer = {y:x for x,y in kmer_inx.items()}

cpdef seq2int_cy(str sequence):
    cdef int L = len(sequence)
    kmer_array = np.zeros(L, dtype=int)
    
    cdef i
    for i in range(l-1,L):
        kmer = sequence[i-l+1:i+1]
        kmer_array[i] = kmer_inx[kmer]
    return kmer_array        


    

cdef extern from "src_helper_avx.c":
    pass
    
cdef extern from "src_helper_avx.h":
    ctypedef struct DerParams:
        c_float_t* za_Ea_derivatives
        c_float_t* zb_Ea_derivatives
        c_float_t* za_Eb_derivatives
        c_float_t* zb_Eb_derivatives
        c_float_t* der_a
        c_float_t* der_b
        
    void initialize_DerParams(DerParams* params, int L, int no_kmers)
    void deinitialize_DerParams(DerParams* params)

    double sum_array_c(double* arr, int length)
    void sum_mat_rows(double* out, double* mat, int n_row, int n_col)
    
    void assign_za_c(int i, double* za, double* zb, double concentration_times_energy, int l)
    void assign_zb_c(long* x, int i, double* za, double* zb, double* Eb, double cab, double sf, double D, double sig, int l)
    
    void assign_za_E_derivatives_c(long* x, int i, double* za, double* zb, int L, int l, int no_kmers,
                                     DerParams* params, double* Ea, double* Eb, double cab)

    void assign_zb_E_derivatives_c(long* x, int i, double* za, double* zb, int L, int l, int no_kmers,
                                     DerParams* params, double* Ea, double* Eb, double cab, double sf, double D, double sig)

    void assign_za_D_derivative_c(int i, double* za_D_derivatives, double* zb_D_derivatives, double concentration_times_energy, int l)

    void assign_za_sig_derivative_c(int i, double* za_sig_derivatives, double* zb_sig_derivatives, double concentration_times_energy, int l)

    void assign_za_sf_derivative_c(int i, double* za_sf_derivatives, double* zb_sf_derivatives, double concentration_times_energy, int l)
    
    void assign_zb_D_derivative_c(int i, double* za, double* za_D_derivatives, double* zb_D_derivatives, double energy_b, 
                                         double cab, double sf, double D , double sig, int l)
    void assign_zb_sig_derivative_c(int i, double* za, double* za_sig_derivatives, double* zb_sig_derivatives, double energy_b, 
                                         double cab, double sf, double D , double sig, int l)
    void assign_zb_sf_derivative_c(int i, double* za, double* za_sf_derivatives, double* zb_sf_derivatives, double energy_b, 
                                         double cab, double sf, double D , double sig, int l)

    double cb_c(int, double, double, double)
    double cb_D_derivative_c(int, double, double, double)
    double cb_sig_derivative_c(int, double, double, double)
    double cb_sf_derivative_c(int, double, double, double)
    

cdef array_from_pointer(void* ptr, int size):
        cdef np.npy_intp shape_c[1]
        shape_c[0] = <np.npy_intp> size
        ndarray = np.PyArray_SimpleNewFromData(1, shape_c, float_dtype, ptr)
        return ndarray
    
    
@cython.boundscheck(False)  # Deactivate bounds checking   
def DP_Z_cy(double[:] args, long[:] x):
    
    cdef int L = len(x)
    cdef int no_kmers = len(kmer_inx)
    cdef double cab = 1.0

    cdef double[:] Ea = args[0:len(kmer_inx)]
    cdef double[:] Eb = args[len(kmer_inx):2*len(kmer_inx)]
    cdef double sf = args[-3]
    cdef double D = args[-2]
    cdef double sig = args[-1]
    
    #initialization of statistical weigths
    cdef double[:] za = np.zeros(L)
    cdef double[:] zb = np.zeros(L)

    cdef int i
    for i in range(0,l):
        zb[i] = 1 
        
        

    #initialization of derivatives
    cdef DerParams params = DerParams()
    initialize_DerParams(&params, L, no_kmers)


    cdef double[:] za_sf_derivatives = np.zeros(L)
    cdef double[:] zb_sf_derivatives = np.zeros(L)
    
    cdef double[:] za_D_derivatives = np.zeros(L)
    cdef double[:] zb_D_derivatives = np.zeros(L)

    cdef double[:] za_sig_derivatives = np.zeros(L)
    cdef double[:] zb_sig_derivatives = np.zeros(L)


    cdef int inx
    
    #precompute (binding energy of domain a binding at position i)*concentration
    cdef double energy_conc_a 
    
    #precompute binding energy of domain b binding at position i
    cdef double energy_b
    
    #dynamic programming calculation of z and derivatives 
    
    for i in range(l,L):
        energy_conc_a = cab*exp(-Ea[x[i]])
        energy_b = exp(-Eb[x[i]])
        
        #calculate statistical weights
        assign_za_c(i, &za[0], &zb[0], energy_conc_a, l)
        assign_zb_c(&x[0], i, &za[0], &zb[0], &Eb[0], cab, sf, D, sig, l)
        
        #calculate derivatives for all kmers (inx) at each position
        assign_za_E_derivatives_c(&x[0], i, &za[0], &zb[0], L, l, len(kmer_inx), 
                                      &params, &Ea[0], &Eb[0], cab)
        assign_zb_E_derivatives_c(&x[0], i, &za[0], &zb[0], L, l, len(kmer_inx), 
                                      &params, &Ea[0], &Eb[0], cab, sf, D, sig)
        
        
        assign_za_sf_derivative_c(i, &za_sf_derivatives[0], &zb_sf_derivatives[0], energy_conc_a, l)
        assign_zb_sf_derivative_c(i, &za[0], &za_sf_derivatives[0], &zb_sf_derivatives[0], energy_b, cab, sf, D, sig, l)
        
        assign_za_D_derivative_c(i, &za_D_derivatives[0], &zb_D_derivatives[0], energy_conc_a, l)
        assign_zb_D_derivative_c(i, &za[0], &za_D_derivatives[0], &zb_D_derivatives[0], energy_b, cab, sf, D, sig, l)
        
        assign_za_sig_derivative_c(i, &za_sig_derivatives[0], &zb_sig_derivatives[0], energy_conc_a, l)
        assign_zb_sig_derivative_c(i, &za[0], &za_sig_derivatives[0], &zb_sig_derivatives[0], energy_b, cab, sf, D, sig, l)

    Z_x = zb[L-1] + sum_array_c(&za[0], L)
    
    #derivative of Z(x)
    
    za_Ea_derivatives = array_from_pointer(params.za_Ea_derivatives, no_kmers * L).reshape(L,-1)
    za_Eb_derivatives = array_from_pointer(params.za_Eb_derivatives, no_kmers * L).reshape(L,-1)
    zb_Ea_derivatives = array_from_pointer(params.zb_Ea_derivatives, no_kmers * L).reshape(L,-1)
    zb_Eb_derivatives = array_from_pointer(params.zb_Eb_derivatives, no_kmers * L).reshape(L,-1)

    d_Ea = zb_Ea_derivatives[L-1,:] + np.sum(za_Ea_derivatives, axis=0)
    d_Eb = zb_Eb_derivatives[L-1,:] + np.sum(za_Eb_derivatives, axis=0)
  
    d_sf = zb_sf_derivatives[L-1] + sum_array_c(&za_sf_derivatives[0], L)
    d_D = zb_D_derivatives[L-1] + sum_array_c(&za_D_derivatives[0], L)
    d_sig = zb_sig_derivatives[L-1] + sum_array_c(&za_sig_derivatives[0], L)       
    gradient = np.concatenate([q.ravel() for q in [d_Ea, d_Eb, np.array([d_sf, d_D, d_sig])]])

    deinitialize_DerParams(&params)
    
    return Z_x, gradient