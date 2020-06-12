# ### cython
from dp_z import *

# ### implementation of the LL object


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