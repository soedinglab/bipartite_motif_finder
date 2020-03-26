
# ### cython

get_ipython().run_cell_magic('cython', '-f -I . --compile-args=-DCYTHON_TRACE=1  --compile-args=-DAVX2=1  --compile-args=-mavx2 ', '\n\ncimport cython\ncimport numpy as np\nctypedef double c_float_t\nimport numpy as np\nimport itertools\nfrom libc.math cimport exp,pow\n\nfloat_dtype = np.NPY_DOUBLE\nnp.import_array()\n\ncdef double cpi = np.pi\ncdef int l = 3 #l_A=l_B=3 nucleotides\n\ncpdef generate_kmer_inx():\n    cdef dict vals = {\'A\':0,\'C\':1,\'G\':2,\'T\':3}\n    cdef dict kmer_inx = {}\n    \n    for p in list(itertools.product(vals.keys(), repeat=l)):\n        inx = 0\n        for j,base in enumerate(p):\n            inx += (4**j)*vals[base] \n        kmer_inx[\'\'.join(p)] = inx\n    return kmer_inx\n\nkmer_inx = generate_kmer_inx()\ninx_kmer = {y:x for x,y in kmer_inx.items()}\n\ncpdef seq2int_cy(str sequence):\n    cdef int L = len(sequence)\n    kmer_array = np.zeros(L, dtype=int)\n    \n    cdef i\n    for i in range(l-1,L):\n        kmer = sequence[i-l+1:i+1]\n        kmer_array[i] = kmer_inx[kmer]\n    return kmer_array        \n\n\n    \n\ncdef extern from "src_helper_avx.c":\n    pass\n    \ncdef extern from "src_helper_avx.h":\n    ctypedef struct DerParams:\n        c_float_t* za_Ea_derivatives\n        c_float_t* zb_Ea_derivatives\n        c_float_t* za_Eb_derivatives\n        c_float_t* zb_Eb_derivatives\n        c_float_t* der_a\n        c_float_t* der_b\n        \n    void initialize_DerParams(DerParams* params, int L, int no_kmers)\n    void deinitialize_DerParams(DerParams* params)\n\n    double sum_array_c(double* arr, int length)\n    void sum_mat_rows(double* out, double* mat, int n_row, int n_col)\n    \n    void assign_za_c(int i, double* za, double* zb, double concentration_times_energy, int l)\n    void assign_zb_c(long* x, int i, double* za, double* zb, double* Eb, double cab, double sf, double D, double sig, int l)\n    \n    void assign_za_E_derivatives_c(long* x, int i, double* za, double* zb, int L, int l, int no_kmers,\n                                     DerParams* params, double* Ea, double* Eb, double cab)\n\n    void assign_zb_E_derivatives_c(long* x, int i, double* za, double* zb, int L, int l, int no_kmers,\n                                     DerParams* params, double* Ea, double* Eb, double cab, double sf, double D, double sig)\n\n    void assign_za_D_derivative_c(int i, double* za_D_derivatives, double* zb_D_derivatives, double concentration_times_energy, int l)\n\n    void assign_za_sig_derivative_c(int i, double* za_sig_derivatives, double* zb_sig_derivatives, double concentration_times_energy, int l)\n\n    void assign_za_sf_derivative_c(int i, double* za_sf_derivatives, double* zb_sf_derivatives, double concentration_times_energy, int l)\n    \n    void assign_zb_D_derivative_c(int i, double* za, double* za_D_derivatives, double* zb_D_derivatives, double energy_b, \n                                         double cab, double sf, double D , double sig, int l)\n    void assign_zb_sig_derivative_c(int i, double* za, double* za_sig_derivatives, double* zb_sig_derivatives, double energy_b, \n                                         double cab, double sf, double D , double sig, int l)\n    void assign_zb_sf_derivative_c(int i, double* za, double* za_sf_derivatives, double* zb_sf_derivatives, double energy_b, \n                                         double cab, double sf, double D , double sig, int l)\n\n    double cb_c(int, double, double, double)\n    double cb_D_derivative_c(int, double, double, double)\n    double cb_sig_derivative_c(int, double, double, double)\n    double cb_sf_derivative_c(int, double, double, double)\n    \n\ncdef array_from_pointer(void* ptr, int size):\n        cdef np.npy_intp shape_c[1]\n        shape_c[0] = <np.npy_intp> size\n        ndarray = np.PyArray_SimpleNewFromData(1, shape_c, float_dtype, ptr)\n        return ndarray\n    \n    \n@cython.boundscheck(False)  # Deactivate bounds checking   \ndef DP_Z_cy(double[:] args, long[:] x):\n    \n    cdef int L = len(x)\n    cdef int no_kmers = len(kmer_inx)\n    cdef double cab = 1.0\n\n    cdef double[:] Ea = args[0:len(kmer_inx)]\n    cdef double[:] Eb = args[len(kmer_inx):2*len(kmer_inx)]\n    cdef double sf = args[-3]\n    cdef double D = args[-2]\n    cdef double sig = args[-1]\n    \n    #initialization of statistical weigths\n    cdef double[:] za = np.zeros(L)\n    cdef double[:] zb = np.zeros(L)\n\n    cdef int i\n    for i in range(0,l):\n        zb[i] = 1 \n        \n        \n\n    #initialization of derivatives\n    cdef DerParams params = DerParams()\n    initialize_DerParams(&params, L, no_kmers)\n\n\n    cdef double[:] za_sf_derivatives = np.zeros(L)\n    cdef double[:] zb_sf_derivatives = np.zeros(L)\n    \n    cdef double[:] za_D_derivatives = np.zeros(L)\n    cdef double[:] zb_D_derivatives = np.zeros(L)\n\n    cdef double[:] za_sig_derivatives = np.zeros(L)\n    cdef double[:] zb_sig_derivatives = np.zeros(L)\n\n\n    cdef int inx\n    \n    #precompute (binding energy of domain a binding at position i)*concentration\n    cdef double energy_conc_a \n    \n    #precompute binding energy of domain b binding at position i\n    cdef double energy_b\n    \n    #dynamic programming calculation of z and derivatives \n    \n    for i in range(l,L):\n        energy_conc_a = cab*exp(-Ea[x[i]])\n        energy_b = exp(-Eb[x[i]])\n        \n        #calculate statistical weights\n        assign_za_c(i, &za[0], &zb[0], energy_conc_a, l)\n        assign_zb_c(&x[0], i, &za[0], &zb[0], &Eb[0], cab, sf, D, sig, l)\n        \n        #calculate derivatives for all kmers (inx) at each position\n        assign_za_E_derivatives_c(&x[0], i, &za[0], &zb[0], L, l, len(kmer_inx), \n                                      &params, &Ea[0], &Eb[0], cab)\n        assign_zb_E_derivatives_c(&x[0], i, &za[0], &zb[0], L, l, len(kmer_inx), \n                                      &params, &Ea[0], &Eb[0], cab, sf, D, sig)\n        \n        \n        assign_za_sf_derivative_c(i, &za_sf_derivatives[0], &zb_sf_derivatives[0], energy_conc_a, l)\n        assign_zb_sf_derivative_c(i, &za[0], &za_sf_derivatives[0], &zb_sf_derivatives[0], energy_b, cab, sf, D, sig, l)\n        \n        assign_za_D_derivative_c(i, &za_D_derivatives[0], &zb_D_derivatives[0], energy_conc_a, l)\n        assign_zb_D_derivative_c(i, &za[0], &za_D_derivatives[0], &zb_D_derivatives[0], energy_b, cab, sf, D, sig, l)\n        \n        assign_za_sig_derivative_c(i, &za_sig_derivatives[0], &zb_sig_derivatives[0], energy_conc_a, l)\n        assign_zb_sig_derivative_c(i, &za[0], &za_sig_derivatives[0], &zb_sig_derivatives[0], energy_b, cab, sf, D, sig, l)\n\n    Z_x = zb[L-1] + sum_array_c(&za[0], L)\n    \n    #derivative of Z(x)\n    \n    za_Ea_derivatives = array_from_pointer(params.za_Ea_derivatives, no_kmers * L).reshape(L,-1)\n    za_Eb_derivatives = array_from_pointer(params.za_Eb_derivatives, no_kmers * L).reshape(L,-1)\n    zb_Ea_derivatives = array_from_pointer(params.zb_Ea_derivatives, no_kmers * L).reshape(L,-1)\n    zb_Eb_derivatives = array_from_pointer(params.zb_Eb_derivatives, no_kmers * L).reshape(L,-1)\n\n    d_Ea = zb_Ea_derivatives[L-1,:] + np.sum(za_Ea_derivatives, axis=0)\n    d_Eb = zb_Eb_derivatives[L-1,:] + np.sum(za_Eb_derivatives, axis=0)\n  \n    d_sf = zb_sf_derivatives[L-1] + sum_array_c(&za_sf_derivatives[0], L)\n    d_D = zb_D_derivatives[L-1] + sum_array_c(&za_D_derivatives[0], L)\n    d_sig = zb_sig_derivatives[L-1] + sum_array_c(&za_sig_derivatives[0], L)       \n    gradient = np.concatenate([q.ravel() for q in [d_Ea, d_Eb, np.array([d_sf, d_D, d_sig])]])\n\n    deinitialize_DerParams(&params)\n    \n    return Z_x, gradient')


# ### implementation of the LL object

# In[6]:


class nLL:
    def __init__(self, seqs_p, seqs_bg):
        
        self.N_p = len(seqs_p)
        self.N_bg = len(seqs_bg)

        #calculate background probabilities:

        #include positive sequences in bg sequences if not there
        X_bg_t = list(set(seqs_p + seqs_bg))  #number of unique sequences
        
        counts = np.zeros(len(X_bg_t))
        for i, x in enumerate(X_bg_t):
            counts[i] = seqs_bg.count(x)
            
        counts = counts + 1 #pseudocount to make sure 
        counts = counts/np.sum(counts)

        p_bg = dict(zip(X_bg_t, counts))

        self.pbg_xp = np.array([p_bg[x] for x in seqs_p])
        self.pbg_xbg = np.array([p_bg[xbg] for xbg in seqs_bg])
        
        #add a padding nucleotide to the beginning to make the calculations stable, binding starts at
        #position i=l so the padded nucleotide has no effect.
        self.X_p = [seq2int_cy('A' + x) for x in seqs_p] 
        self.X_bg = [seq2int_cy('A' + x) for x in seqs_bg]

        
    def assign_z_p(self, tup):
            i, args = tup
            d_z_x_np = np.frombuffer(dz.get_obj(), dtype=np.float64).reshape(-1, self.N_p)
            z[i], d_z_x_np[:,i] = DP_Z_cy(args, self.X_p[i])
            
    def assign_z_bg(self, tup):
            i, args = tup
            d_z_xbg_np = np.frombuffer(dz.get_obj(), dtype=np.float64).reshape(-1, self.N_bg)
            z[i], d_z_xbg_np[:,i] = DP_Z_cy(args, self.X_bg[i])

          

        
    def __call__(self, parameters):
        
        #number of positive variables (stacked at the end)
        n_pos = 3
        
        #exp parameters to make sure they are positive
        args = parameters.copy()
        args[-n_pos:] = np.exp(args[-n_pos:])
    
    
        #define weights and derivatives as a multiprocessing array
        z_x = mp.Array(ctypes.c_double, self.N_p)
        d_z_x = mp.Array(ctypes.c_double, (2*len(kmer_inx)+ n_pos)*self.N_p)

        z_xbg = mp.Array(ctypes.c_double, self.N_bg)
        d_z_xbg = mp.Array(ctypes.c_double, (2*len(kmer_inx)+ n_pos)*self.N_bg) 
        
        #parallelizing
        with multiprocessing.Pool(initializer=init, initargs=(z_x,d_z_x), processes=8) as pool:
            pool.map(self.assign_z_p, [(i, args) for i in range(len(self.X_p))])
        with multiprocessing.Pool(initializer=init, initargs=(z_xbg, d_z_xbg), processes=8)  as pool:
            pool.map(self.assign_z_bg, [(i, args) for i in range(len(self.X_bg))])
        
        #= convert to np array ======
        d_z_x = np.frombuffer(d_z_x.get_obj(), dtype=np.float64).reshape(-1, self.N_p)
        d_z_xbg = np.frombuffer(d_z_xbg.get_obj(), dtype=np.float64).reshape(-1, self.N_bg)
        z_x = np.frombuffer(z_x.get_obj(), dtype=np.float64)
        z_xbg = np.frombuffer(z_xbg.get_obj(), dtype=np.float64)
        #============================
        
        #calculate log likelihood of model given arg parameters
        ll = np.sum(np.log(self.pbg_xp) + np.log(np.ones(self.N_p) - (np.ones(self.N_p)/z_x)))
        ll -= self.N_p * logsumexp( np.log(self.pbg_xbg) + np.log(np.ones(self.N_bg) - (np.ones(self.N_bg)/z_xbg)) )
        
        #calculate partial derivatives of model given arg parameters
        dll = np.sum(d_z_x/(z_x*(z_x-1)), axis=1)
        dll -= self.N_p * ( np.sum((self.pbg_xbg * d_z_xbg)/(z_xbg*z_xbg), axis=1 ) / np.sum(self.pbg_xbg*(np.ones(self.N_bg) - (np.ones(self.N_bg)/z_xbg))))

        #exp modify dLL for positive elements
        dll[-n_pos:] = dll[-n_pos:]*args[-n_pos:]

        #regularize some parameters
        if False:
            comp = -3
            reg = 1e-8 
            ll -= np.power(args[comp],2)*reg
            dll[comp] -= 2*reg*args[comp]
        return -ll, -dll 

#make the arrays global to all processes
def init(z_array, dz_array):
    global z
    z = z_array    
    global dz
    dz = dz_array
