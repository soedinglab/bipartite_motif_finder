#include "src_helper_avx.h"
#include "../lib/simd.h"

static inline void initialize_array(c_float_t* arr, c_float_t value, int length) {
  for(int i = 0; i < length; i++) {
    arr[i] = value;
  }
}

static inline void add_array(c_float_t* out, c_float_t* x, c_float_t* y, size_t N) {
  for (int n = 0; n < N; n+=VECSIZE_DOUBLE) {
    simd_double x_chunk = simdf64_load(x + n);
    simd_double y_chunk = simdf64_load(y + n);
    simdf64_store(out + n,  simdf64_add(x_chunk, y_chunk));
  }
}

//out = x + y*constant
static inline void add_mul_constant(c_float_t* out, c_float_t* x, c_float_t* y, c_float_t constant, size_t N) {
  simd_double const_chunk = simdf64_set(constant);
  for (int n = 0; n < N; n+=VECSIZE_DOUBLE) {
    simd_double y_chunk = simdf64_load(y + n);

    simd_double x_chunk = simdf64_load(x + n);
    simd_double mul_yc = simdf64_mul(y_chunk, const_chunk);

    simdf64_store(out + n,  simdf64_add(x_chunk, mul_yc) );
  }
}

//out = x*constant
static inline void mul_constant(c_float_t* out, c_float_t* x, c_float_t constant, size_t N) {
  simd_double const_chunk = simdf64_set(constant);
  for (int n = 0; n < N; n+=VECSIZE_DOUBLE) {
    simd_double x_chunk = simdf64_load(x + n);
    simd_double mul_xc = simdf64_mul(x_chunk, const_chunk);

    simdf64_store(out + n,  mul_xc);
  }
}



double sum_array_c(double* arr, int length)
{
    double sum = 0;
    for(int i = 0; i < length; i++) 
    {
        sum += arr[i];
    }
    return sum;
}

void sum_mat_rows(double* out, double* mat, int n_row, int n_col)
{
    for (int j=0; j<n_col; j++)
    {
        out[j] = 0;
        for (int i=0; i<n_row; i++)
        {
            out[j] += mat[i*n_col + j];
        }
    }
}

void initialize_DerParams(DerParams* params, int L, int no_kmers) {

    params->der_a = malloc_simd_double(sizeof(c_float_t) * no_kmers);
    initialize_array(params->der_a, 0, no_kmers);

    params->der_b = malloc_simd_double(sizeof(c_float_t) * no_kmers);
    initialize_array(params->der_b, 0, no_kmers);

    params->za_Ea_derivatives = malloc_simd_double(sizeof(c_float_t) * no_kmers * L);
    initialize_array(params->za_Ea_derivatives, 0, no_kmers * L);

    params->za_Eb_derivatives = malloc_simd_double(sizeof(c_float_t) * no_kmers * L);
    initialize_array(params->za_Eb_derivatives, 0, no_kmers * L);

    params->zb_Ea_derivatives = malloc_simd_double(sizeof(c_float_t) * no_kmers * L);
    initialize_array(params->zb_Ea_derivatives, 0, no_kmers * L);

    params->zb_Eb_derivatives = malloc_simd_double(sizeof(c_float_t) * no_kmers * L);
    initialize_array(params->zb_Eb_derivatives, 0, no_kmers * L);


}

void deinitialize_DerParams(DerParams* params) {

    free(params->der_a);
    free(params->der_b);
    free(params->za_Ea_derivatives);
    free(params->za_Eb_derivatives);
    free(params->zb_Ea_derivatives);
    free(params->zb_Eb_derivatives);

}

void assign_za_c(int i, double* za, double* zb, double concentration_times_energy, int l)
{
    double za_tmp = zb[i-l];
    for (int j=0; j<i-l+1; j++)
    {
        za_tmp += za[j];
    }
    za[i] = (za_tmp)*concentration_times_energy;
}

void assign_zb_c(long* x, int i, double* za, double* zb, double* Eb, double cab, double sf, double D, double sig, int l)
{    
    double zb_temp = 0;
    double energy = exp(-Eb[x[i]]);
    for (int j=0; j<i-l+1; j++)
    {
        zb_temp += za[j]*cb_c(i-j-l, sf, D, sig);
    }
    zb_temp *= energy;
    zb_temp += zb[i-l]*cab*energy;      
    zb[i] = zb[i-1] + zb_temp;
}


void assign_za_E_derivatives_c(long* x, int i, double* za, double* zb, int L, int l, int no_kmers,
                                 DerParams* params, double* Ea, double* Eb, double cab)
{ 
    double* za_Ea_derivatives = params->za_Ea_derivatives;
    double* zb_Ea_derivatives = params->zb_Ea_derivatives;
    double* za_Eb_derivatives = params->za_Eb_derivatives;
    double* zb_Eb_derivatives = params->zb_Eb_derivatives;
    double* der_a = params->der_a;
    double* der_b = params->der_b;

    double energy = exp(-Ea[x[i]])*cab;
    for (int inx=0; inx<no_kmers; inx++)
    {  
        der_a[inx] = zb_Ea_derivatives[(i-l)*no_kmers + inx]; 
        der_b[inx] = zb_Eb_derivatives[(i-l)*no_kmers + inx];
    }
    der_a[x[i]] -= zb[i-l];

    for (int j=0; j<i-l+1; ++j)
    {
        add_array(der_a, der_a, za_Ea_derivatives + j*no_kmers, no_kmers);
        add_array(der_b, der_b, za_Eb_derivatives + j*no_kmers, no_kmers);
        der_a[x[i]] -= za[j];
    }

    mul_constant(za_Ea_derivatives + i*no_kmers, der_a, energy, no_kmers);
    mul_constant(za_Eb_derivatives + i*no_kmers, der_b, energy, no_kmers);
}


void assign_zb_E_derivatives_c(long* x, int i, double* za, double* zb, int L, int l, int no_kmers,
                                 DerParams* params, double* Ea, double* Eb, double cab, double sf, double D, double sig)
 {
    double* za_Ea_derivatives = params->za_Ea_derivatives;
    double* zb_Ea_derivatives = params->zb_Ea_derivatives;
    double* za_Eb_derivatives = params->za_Eb_derivatives;
    double* zb_Eb_derivatives = params->zb_Eb_derivatives;
    double* der_a = params->der_a;
    double* der_b = params->der_b;

    double energy = exp(-Eb[x[i]]);
    for (int inx=0; inx<no_kmers; inx++)
    {   
        der_b[inx] = zb_Eb_derivatives[(i-1)*no_kmers + inx];
        der_a[inx] = zb_Ea_derivatives[(i-1)*no_kmers + inx];
    }
  
    for (int j=0; j<i-l+1; ++j)
    {
        double concentration_times_energy = cb_c(i-j-l, sf, D, sig) * energy;
        add_mul_constant(der_b, der_b, za_Eb_derivatives + j*no_kmers, concentration_times_energy, no_kmers);
        add_mul_constant(der_a, der_a, za_Ea_derivatives + j*no_kmers, concentration_times_energy, no_kmers);

        der_b[x[i]] -= concentration_times_energy*za[j];

    }

    double conc = energy*cab;
    add_mul_constant(der_b, der_b, zb_Eb_derivatives + (i-l)*no_kmers, conc, no_kmers);
    add_mul_constant(der_a, der_a, zb_Ea_derivatives + (i-l)*no_kmers, conc, no_kmers);

    der_b[x[i]] -= cab * zb[i-l]*energy;

    for (int inx=0; inx<no_kmers; inx++)
    {
        zb_Ea_derivatives[i*no_kmers + inx] = der_a[inx];
        zb_Eb_derivatives[i*no_kmers + inx] = der_b[inx];
    }
 }

void assign_za_D_derivative_c(int i, double* za_D_derivatives, double* zb_D_derivatives, double concentration_times_energy, int l)
{
     double der_tmp = 0;
    for (int j=0; j<i-l+1; j++)
    {
        der_tmp += za_D_derivatives[j];
    }
    za_D_derivatives[i] = (zb_D_derivatives[i-l] + der_tmp)*concentration_times_energy;
}    


void assign_za_sig_derivative_c(int i, double* za_sig_derivatives, double* zb_sig_derivatives, double concentration_times_energy, int l)
 {
    double der_tmp = 0;
    for (int j=0; j<i-l+1; j++)
    {
        der_tmp += za_sig_derivatives[j];
    }
    za_sig_derivatives[i] = (zb_sig_derivatives[i-l] + der_tmp)*concentration_times_energy;
} 


void assign_za_sf_derivative_c(int i, double* za_sf_derivatives, double* zb_sf_derivatives, double concentration_times_energy, int l)
 {
    double der_tmp = 0;
    for (int j=0; j<i-l+1; j++)
    {
        der_tmp += za_sf_derivatives[j];
    }
    za_sf_derivatives[i] = (zb_sf_derivatives[i-l] + der_tmp)*concentration_times_energy;
}


void assign_zb_D_derivative_c(int i, double* za, double* za_D_derivatives, double* zb_D_derivatives, double energy_b, 
                                     double cab, double sf, double D , double sig, int l)
{    
    double der_tmp = 0;
    for (int j=0; j<i-l+1; j++)
    {
        der_tmp += za_D_derivatives[j]*cb_c(i-l-j, sf, D, sig) + za[j]*cb_D_derivative_c(i-l-j, sf, D, sig);
    }
    der_tmp += zb_D_derivatives[i-l]*cab;
    der_tmp *= energy_b;
    der_tmp += zb_D_derivatives[i-1];
    
    zb_D_derivatives[i] = der_tmp;
 }   
void assign_zb_sig_derivative_c(int i, double* za, double* za_sig_derivatives, double* zb_sig_derivatives, double energy_b, 
                                     double cab, double sf, double D , double sig, int l)
{
    double der_tmp = 0;
    for (int j=0; j<i-l+1; j++)
    {
        der_tmp += za_sig_derivatives[j]*cb_c(i-l-j, sf, D, sig) + za[j]*cb_sig_derivative_c(i-l-j, sf, D, sig);
    }
    der_tmp += zb_sig_derivatives[i-l]*cab;
    der_tmp *= energy_b;
    der_tmp += zb_sig_derivatives[i-1];
    
    zb_sig_derivatives[i] = der_tmp;
}


void assign_zb_sf_derivative_c(int i, double* za, double* za_sf_derivatives, double* zb_sf_derivatives, double energy_b, 
                                     double cab, double sf, double D , double sig, int l)
{
    double der_tmp = 0;
    for (int j=0; j<i-l+1; j++)
    {
        der_tmp += za_sf_derivatives[j]*cb_c(i-l-j, sf, D, sig) + za[j]*cb_sf_derivative_c(i-l-j, sf, D, sig);
    }
    der_tmp += zb_sf_derivatives[i-l]*cab;
    der_tmp *= energy_b;
    der_tmp += zb_sf_derivatives[i-1];
    
    zb_sf_derivatives[i] = der_tmp;
}


double inline cb_c(int d, double sf, double D, double sig)
{   
    if (d < 0)
        return 0;
    
    double diff = d - D; 
    double gaussian = exp(- (diff*diff) / (2.0 *(sig*sig )));

    return gaussian*sf + 1 ;
}

double cb_D_derivative_c(int d, double sf, double D, double sig)
{    
    if (d < 0)
        return 0;

    double diff = d - D; 
    double sig2 = sig*sig;
    double der = sf*diff*exp(-diff*diff/(2.0 *sig2))/(sig2);
    return der;
    
}

double cb_sig_derivative_c(int d, double sf, double D, double sig)
{    
    if (d < 0)
        return 0;

    double diff = d - D;
    double sig2 = sig*sig; 
    double der = diff*diff*sig/((sig2)*(sig2));
    der *= sf*exp(-diff*diff/(2.0 *(sig2)));
    return der;
    
}

double cb_sf_derivative_c(int d, double sf, double D, double sig)
{    
    if (d < 0)
        return 0;

    double diff = d - D; 
    double gaussian = exp(- (diff*diff) / (2.0 *(sig*sig)));
    return gaussian;
    
}
