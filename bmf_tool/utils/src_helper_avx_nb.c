#include "src_helper_avx_nb.h"
#include "simd.h"
#include <stdio.h>
#include <stdlib.h>


static inline void initialize_array(c_float_t* arr, c_float_t value, int length) {
  for(int i = 0; i < length; i++) {
    arr[i] = value;
  }
}

#ifdef USEAVX_DOUBLE

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
    

#elif USEAVX_FLOAT

    static inline void add_array(c_float_t* out, c_float_t* x, c_float_t* y, size_t N) {
        for (int n = 0; n < N; n+=VECSIZE_FLOAT) {
            simd_float x_chunk = simdf32_load(x + n);
            simd_float y_chunk = simdf32_load(y + n);
            simdf32_store(out + n,  simdf32_add(x_chunk, y_chunk));
        }
    }

    //out = x + y*constant
    static inline void add_mul_constant(c_float_t* out, c_float_t* x, c_float_t* y, c_float_t constant, size_t N) {
        simd_float const_chunk = simdf32_set(constant);
        for (int n = 0; n < N; n+=VECSIZE_FLOAT) {
            simd_float y_chunk = simdf32_load(y + n);

            simd_float x_chunk = simdf32_load(x + n);
            simd_float mul_yc = simdf32_mul(y_chunk, const_chunk);

            simdf32_store(out + n,  simdf32_add(x_chunk, mul_yc) );
        }
    }

    //out = x*constant
    static inline void mul_constant(c_float_t* out, c_float_t* x, c_float_t constant, size_t N) {
        simd_float const_chunk = simdf32_set(constant);
        for (int n = 0; n < N; n+=VECSIZE_FLOAT) {
            simd_float x_chunk = simdf32_load(x + n);
            simd_float mul_xc = simdf32_mul(x_chunk, const_chunk);

            simdf32_store(out + n,  mul_xc);
        }
    }


#else

    //out = x + y
    static inline void add_array(c_float_t* out, c_float_t* x, c_float_t* y, size_t N) {
        for (size_t n = 0; n < N; n+=1) {
            out[n] = x[n] + y[n];
        }
    }

    //out = x + y*constant
    static inline void add_mul_constant(c_float_t* out, c_float_t* x, c_float_t* y, c_float_t constant, size_t N) {
        for (size_t n = 0; n < N; n+=1) {
            out[n] = x[n] + (y[n]*constant);
        }
    }

    //out = x*constant
    static inline void mul_constant(c_float_t* out, c_float_t* x, c_float_t constant, size_t N) {
        for (size_t n = 0; n < N; n+=1) {
            out[n] = x[n]*constant;
        }
    }

#endif

//for double definitions
#if defined(USEAVX_DOUBLE) || defined(NOAVX_DOUBLE) 

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

#else

    void initialize_DerParams(DerParams* params, int L, int no_kmers) {

        params->der_a = malloc_simd_float(sizeof(c_float_t) * no_kmers);
        initialize_array(params->der_a, 0, no_kmers);

        params->der_b = malloc_simd_float(sizeof(c_float_t) * no_kmers);
        initialize_array(params->der_b, 0, no_kmers);

        params->za_Ea_derivatives = malloc_simd_float(sizeof(c_float_t) * no_kmers * L);
        initialize_array(params->za_Ea_derivatives, 0, no_kmers * L);

        params->za_Eb_derivatives = malloc_simd_float(sizeof(c_float_t) * no_kmers * L);
        initialize_array(params->za_Eb_derivatives, 0, no_kmers * L);

        params->zb_Ea_derivatives = malloc_simd_float(sizeof(c_float_t) * no_kmers * L);
        initialize_array(params->zb_Ea_derivatives, 0, no_kmers * L);

        params->zb_Eb_derivatives = malloc_simd_float(sizeof(c_float_t) * no_kmers * L);
        initialize_array(params->zb_Eb_derivatives, 0, no_kmers * L);
    }
#endif


c_float_t sum_array_c(c_float_t* arr, int length)
{
    c_float_t sum = 0;
    for(int i = 0; i < length; i++) 
    {
        sum += arr[i];
    }
    return sum;
}

void sum_mat_rows(c_float_t* out, c_float_t* mat, int n_row, int n_col)
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

void deinitialize_DerParams(DerParams* params) {

    free(params->der_a);
    free(params->der_b);
    free(params->za_Ea_derivatives);
    free(params->za_Eb_derivatives);
    free(params->zb_Ea_derivatives);
    free(params->zb_Eb_derivatives);

}

void assign_za_c(int i, c_float_t* za, c_float_t* zb, c_float_t concentration_times_energy, int l)
{
    c_float_t za_tmp = zb[i-l];
    for (int j=0; j<i-l+1; j++)
    {
        za_tmp += za[j];
    }
    za[i] = (za_tmp)*concentration_times_energy;
}

void assign_zb_c(long* x, int i, c_float_t* za, c_float_t* zb, c_float_t* Eb, c_float_t cab, c_float_t sf, c_float_t r, c_float_t p, int l)
{    
    c_float_t zb_temp = 0;
    c_float_t energy = exp(-Eb[x[i]]);
    for (int j=0; j<i-l+1; j++)
    {
        zb_temp += za[j]*cb_c(i-j-l, sf, r, p);
    }
    zb_temp *= energy;
    zb_temp += zb[i-l]*cab*energy;      
    zb[i] = zb[i-1] + zb_temp;
}


void assign_za_E_derivatives_c(long* x, int i, c_float_t* za, c_float_t* zb, int L, int l, int no_kmers,
                                 DerParams* params, c_float_t* Ea, c_float_t* Eb, c_float_t cab)
{ 
    c_float_t* za_Ea_derivatives = params->za_Ea_derivatives;
    c_float_t* zb_Ea_derivatives = params->zb_Ea_derivatives;
    c_float_t* za_Eb_derivatives = params->za_Eb_derivatives;
    c_float_t* zb_Eb_derivatives = params->zb_Eb_derivatives;
    c_float_t* der_a = params->der_a;
    c_float_t* der_b = params->der_b;

    c_float_t energy = exp(-Ea[x[i]])*cab;
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


void assign_zb_E_derivatives_c(long* x, int i, c_float_t* za, c_float_t* zb, int L, int l, int no_kmers,
                                 DerParams* params, c_float_t* Ea, c_float_t* Eb, c_float_t cab, c_float_t sf, c_float_t r, c_float_t p)
 {
    c_float_t* za_Ea_derivatives = params->za_Ea_derivatives;
    c_float_t* zb_Ea_derivatives = params->zb_Ea_derivatives;
    c_float_t* za_Eb_derivatives = params->za_Eb_derivatives;
    c_float_t* zb_Eb_derivatives = params->zb_Eb_derivatives;
    c_float_t* der_a = params->der_a;
    c_float_t* der_b = params->der_b;

    c_float_t energy = exp(-Eb[x[i]]);
    for (int inx=0; inx<no_kmers; inx++)
    {   
        der_b[inx] = zb_Eb_derivatives[(i-1)*no_kmers + inx];
        der_a[inx] = zb_Ea_derivatives[(i-1)*no_kmers + inx];
    }
  
    for (int j=0; j<i-l+1; ++j)
    {
        c_float_t concentration_times_energy = cb_c(i-j-l, sf, r, p) * energy;
        add_mul_constant(der_b, der_b, za_Eb_derivatives + j*no_kmers, concentration_times_energy, no_kmers);
        add_mul_constant(der_a, der_a, za_Ea_derivatives + j*no_kmers, concentration_times_energy, no_kmers);

        der_b[x[i]] -= concentration_times_energy*za[j];

    }

    c_float_t conc = energy*cab;
    add_mul_constant(der_b, der_b, zb_Eb_derivatives + (i-l)*no_kmers, conc, no_kmers);
    add_mul_constant(der_a, der_a, zb_Ea_derivatives + (i-l)*no_kmers, conc, no_kmers);

    der_b[x[i]] -= cab * zb[i-l]*energy;

    for (int inx=0; inx<no_kmers; inx++)
    {
        zb_Ea_derivatives[i*no_kmers + inx] = der_a[inx];
        zb_Eb_derivatives[i*no_kmers + inx] = der_b[inx];
    }
 }

void assign_za_r_derivative_c(int i, c_float_t* za_r_derivatives, c_float_t* zb_r_derivatives, c_float_t concentration_times_energy, int l)
{
     c_float_t der_tmp = 0;
    for (int j=0; j<i-l+1; j++)
    {
        der_tmp += za_r_derivatives[j];
    }
    za_r_derivatives[i] = (zb_r_derivatives[i-l] + der_tmp)*concentration_times_energy;
}    


void assign_za_p_derivative_c(int i, c_float_t* za_p_derivatives, c_float_t* zb_p_derivatives, c_float_t concentration_times_energy, int l)
 {
    c_float_t der_tmp = 0;
    for (int j=0; j<i-l+1; j++)
    {
        der_tmp += za_p_derivatives[j];
    }
    za_p_derivatives[i] = (zb_p_derivatives[i-l] + der_tmp)*concentration_times_energy;
} 


void assign_za_sf_derivative_c(int i, c_float_t* za_sf_derivatives, c_float_t* zb_sf_derivatives, c_float_t concentration_times_energy, int l)
 {
    c_float_t der_tmp = 0;
    for (int j=0; j<i-l+1; j++)
    {
        der_tmp += za_sf_derivatives[j];
    }
    za_sf_derivatives[i] = (zb_sf_derivatives[i-l] + der_tmp)*concentration_times_energy;
}


void assign_zb_r_derivative_c(int i, c_float_t* za, c_float_t* za_r_derivatives, c_float_t* zb_r_derivatives, c_float_t energy_b, 
                                     c_float_t cab, c_float_t sf, c_float_t r, c_float_t p, int l)
{    
    c_float_t der_tmp = 0;
    for (int j=0; j<i-l+1; j++)
    {
        der_tmp += za_r_derivatives[j]*cb_c(i-l-j, sf, r, p) + za[j]*cb_r_derivative_c(i-l-j, sf, r, p);
    }
    der_tmp += zb_r_derivatives[i-l]*cab;
    der_tmp *= energy_b;
    der_tmp += zb_r_derivatives[i-1];
    
    zb_r_derivatives[i] = der_tmp;
 }   
void assign_zb_p_derivative_c(int i, c_float_t* za, c_float_t* za_p_derivatives, c_float_t* zb_p_derivatives, c_float_t energy_b, 
                                     c_float_t cab, c_float_t sf, c_float_t r, c_float_t p, int l)
{
    c_float_t der_tmp = 0;
    for (int j=0; j<i-l+1; j++)
    {
        der_tmp += za_p_derivatives[j]*cb_c(i-l-j, sf, r, p) + za[j]*cb_p_derivative_c(i-l-j, sf, r, p);
    }
    der_tmp += zb_p_derivatives[i-l]*cab;
    der_tmp *= energy_b;
    der_tmp += zb_p_derivatives[i-1];
    
    zb_p_derivatives[i] = der_tmp;
}


void assign_zb_sf_derivative_c(int i, c_float_t* za, c_float_t* za_sf_derivatives, c_float_t* zb_sf_derivatives, c_float_t energy_b, 
                                     c_float_t cab, c_float_t sf, c_float_t r, c_float_t p, int l)
{
    c_float_t der_tmp = 0;
    for (int j=0; j<i-l+1; j++)
    {
        der_tmp += za_sf_derivatives[j]*cb_c(i-l-j, sf, r, p) + za[j]*cb_sf_derivative_c(i-l-j, sf, r, p);
    }
    der_tmp += zb_sf_derivatives[i-l]*cab;
    der_tmp *= energy_b;
    der_tmp += zb_sf_derivatives[i-1];
    
    zb_sf_derivatives[i] = der_tmp;
}


c_float_t inline cb_c(int d, c_float_t sf, c_float_t r, c_float_t p)
{   
    if (d < 0)
        return 0;

    c_float_t lognb = lgamma(d+r) + d*log(1-p) + r*log(p) - lgamma(r) - lgamma(d+1);
    return exp(lognb)*sf + 1 ;
}

c_float_t cb_r_derivative_c(int d, c_float_t sf, c_float_t r, c_float_t p)
{    
    if (d < 0)
        return 0;
    
    c_float_t der = digamma(d+r) + log(p) - digamma(r);
    c_float_t lognb = lgamma(d+r) + d*log(1-p) + r*log(p) - lgamma(r) - lgamma(d+1);
    return der*exp(lognb)*sf;
    
}

c_float_t cb_p_derivative_c(int d, c_float_t sf, c_float_t r, c_float_t p)
{    
    if (d < 0)
        return 0;

    c_float_t der = -d/(1-p) + r/p;
    c_float_t lognb = lgamma(d+r) + d*log(1-p) + r*log(p) - lgamma(r) - lgamma(d+1);
    return der*exp(lognb)*sf;
    
}


c_float_t cb_sf_derivative_c(int d, c_float_t sf, c_float_t r, c_float_t p)
{    
    if (d < 0)
        return 0;

    c_float_t lognb = lgamma(d+r) + d*log(1-p) + r*log(p) - lgamma(r) - lgamma(d+1);
    return exp(lognb);
    
}



/* digamma.c
 *
 * Mark Johnson, 2nd September 2007
 *
 * Computes the Ψ(x) or digamma function, i.e., the derivative of the 
 * log gamma function, using a series expansion.
 *
 * Warning:  I'm not a numerical analyst, so I may have made errors here!
 *
 * The parameters of the series were computed using the Maple symbolic
 * algebra program as follows:
 *
 * series(Psi(x+1/2), x=infinity, 21);
 *
 * which produces:
 *
 *  ln(x)+1/(24*x^2)-7/960/x^4+31/8064/x^6-127/30720/x^8+511/67584/x^10-1414477/67092480/x^12+8191/98304/x^14-118518239/267386880/x^16+5749691557/1882718208/x^18-91546277357/3460300800/x^20+O(1/(x^21)) 
 *
 * It looks as if the terms in this expansion *diverge* as the powers
 * get larger.  However, for large x, the x^-n term will dominate.
 *
 * I used Maple to examine the difference between this series and
 * Digamma(x+1/2) over the range 7 < x < 20, and discovered that the
 * difference is less that 1e-8 if the terms up to x^-8 are included.
 * This determined the power used in the code here.  Of course,
 * Maple uses some kind of approximation to calculate Digamma,
 * so all I've really done here is find the smallest power that produces
 * the Maple approximation; still, that should be good enough for our
 * purposes.
 *
 * This expansion is accurate for x > 7; we use the recurrence 
 *
 * digamma(x) = digamma(x+1) - 1/x
 *
 * to make x larger than 7.
 */

c_float_t digamma(c_float_t x) {
  c_float_t result = 0, xx, xx2, xx4;
  assert(x > 0);
  for ( ; x < 7; ++x)
    result -= 1/x;
  x -= 1.0/2.0;
  xx = 1.0/x;
  xx2 = xx*xx;
  xx4 = xx2*xx2;
  result += log(x)+(1./24.)*xx2-(7.0/960.0)*xx4+(31.0/8064.0)*xx4*xx2-(127.0/30720.0)*xx4*xx4;
  return result;
}