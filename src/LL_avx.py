# ### cython
from dp_z import *

# ### implementation of the LL object


class nLL:
    def __init__(self, seqs_p, seqs_bg, core_length):
        
        self.N_p = len(seqs_p)
        self.N_bg = len(seqs_bg)
        
        self.core_length = core_length
        
        #calculate background probabilities:

        #include positive sequences in bg sequences if not there
        X_bg_t = list(set(np.concatenate([[seqs_p[:,0],seqs_bg[:,0]]]).ravel()))  #number of unique sequences
        
        #dictionary with sequences as keys and their frequencies as values
        no_sequences = len(X_bg_t)
        p_bg = {k:v/no_sequences for k,v in Counter(X_bg_t).items()}


        self.pbg_xp = np.array([p_bg[x] for x in seqs_p[:,0]])
        self.pbg_xbg = np.array([p_bg[xbg] for xbg in seqs_bg[:,0]])
        
        
        #create sequence to integer dictionaries for the RNA sequence and structure
        kmer_inx = generate_kmer_inx(core_length)
        struct_inx = generate_struct_inx(core_length)
        
        #I add a padding nucleotide to the beginning to avoid -1 indexing, binding starts at
        #position i=l so the padded nucleotide has no effect.
        self.X_p  = [seq2int_cy('A' + x , core_length, kmer_inx) for x in seqs_p[:,0]] 
        self.X_bg = [seq2int_cy('A' + x , core_length, kmer_inx) for x in seqs_bg[:,0]]
        self.S_p  = [seq2int_cy('.' + x , core_length, struct_inx) for x in seqs_p[:,1]] 
        self.S_bg = [seq2int_cy('.' + x , core_length, struct_inx) for x in seqs_bg[:,1]]


        
    def assign_z_p(self, tup):
            i, args = tup
            d_z_x_np = np.frombuffer(dz.get_obj(), dtype=np.float64).reshape(-1, self.N_p)
            z[i], d_z_x_np[:,i] = DP_Z_cy(args, self.X_p[i], self.S_p[i], self.core_length)
                
            
    def assign_z_bg(self, tup):
            i, args = tup
            d_z_xbg_np = np.frombuffer(dz.get_obj(), dtype=np.float64).reshape(-1, self.N_bg)
            z[i], d_z_xbg_np[:,i] = DP_Z_cy(args, self.X_bg[i], self.S_bg[i], self.core_length)

          

        
    def __call__(self, parameters):
        
        #number of processes
        n_processes = 4
        
        #number of model parameters
        n_param = parameters.shape[0]
        
        #number of positive variables (stacked at the end)
        n_pos = 3
        
        #exp parameters to make sure they are positive
        args = parameters.copy()
        args[-n_pos:-1] = np.exp(args[-n_pos:-1])
        exp_p = np.exp(-args[-1])
        
        #the last parameter is p and should stay between 0 and 1
        args[-1] = 1/(1+exp_p)
    
        #define weights and derivatives as a multiprocessing array
        z_x = mp.Array(ctypes.c_double, self.N_p)
        d_z_x = mp.Array(ctypes.c_double, n_param*self.N_p)

        z_xbg = mp.Array(ctypes.c_double, self.N_bg)
        d_z_xbg = mp.Array(ctypes.c_double, n_param*self.N_bg) 
        
        #parallelizing
        with multiprocessing.Pool(initializer=init, initargs=(z_x,d_z_x), processes=n_processes) as pool:
            pool.map(self.assign_z_p, [(i, args) for i in range(len(self.X_p))])
        with multiprocessing.Pool(initializer=init, initargs=(z_xbg, d_z_xbg), processes=n_processes)  as pool:
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
        dll[-n_pos:-1] = dll[-n_pos:-1]*args[-n_pos:-1]
        dll[-1] = dll[-1]*args[-1]*(1-args[-1])

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
