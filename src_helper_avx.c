#include "src_helper_avx.h"

void assign_za_E_derivatives_c(long* x, int i, double* za, double* zb, int L, int l, int no_kmers,
                                 double* za_Ea_derivatives, double* zb_Ea_derivatives, double* za_Eb_derivatives, double* zb_Eb_derivatives,
                                 double* Ea, double* Eb, double cab, double* der_a, double* der_b)
{  
    double energy = exp(-Ea[x[i]])*cab;
    for (int inx=0; inx<no_kmers; inx++)
    {  
        der_a[inx] = zb_Ea_derivatives[(i-l)*no_kmers + inx]; 
        der_b[inx] = zb_Eb_derivatives[(i-l)*no_kmers + inx];
    }
    der_a[x[i]] -= zb[i-l];

    for (int j=0; j<i-l+1; ++j)
    {
        for (int inx=0; inx<no_kmers; inx++)
        {  
            der_a[inx] += za_Ea_derivatives[j*no_kmers + inx];
            der_b[inx] += za_Eb_derivatives[j*no_kmers + inx];
        }
        der_a[x[i]] -= za[j];
    }
    for (int inx=0; inx<no_kmers; inx++)
    {
        za_Ea_derivatives[i*no_kmers + inx] = der_a[inx]*energy;
        za_Eb_derivatives[i*no_kmers + inx] = der_b[inx]*energy;
    }
}

void assign_zb_E_derivatives_c(long* x, int i, double* za, double* zb, int L, int l, int no_kmers,
                                 double* za_Ea_derivatives, double* zb_Ea_derivatives, double* za_Eb_derivatives, double* zb_Eb_derivatives,
                                 double* Ea, double* Eb, double cab, double sf, double D, double sig, double* der_a, double* der_b)
 {
    double energy = exp(-Eb[x[i]]);
    for (int inx=0; inx<no_kmers; inx++)
    {   
        der_b[inx] = zb_Eb_derivatives[(i-1)*no_kmers + inx];
        der_a[inx] = zb_Ea_derivatives[(i-1)*no_kmers + inx];
    }
  
    for (int j=0; j<i-l+1; ++j)
    {
        double concentration_j = cb_c(i-j-l, sf, D, sig);
        for (int inx=0; inx<no_kmers; inx++)
        {
            der_b[inx] += concentration_j * za_Eb_derivatives[j*no_kmers + inx] * energy;
            der_a[inx] += concentration_j * za_Ea_derivatives[j*no_kmers + inx] * energy;  
        }
        der_b[x[i]] -= concentration_j*za[j]*energy;

    }

    for (int inx=0; inx<no_kmers; inx++)
    {
        der_b[inx] += cab * zb_Eb_derivatives[(i-l)*no_kmers + inx] * energy; 
        der_a[inx] += cab * zb_Ea_derivatives[(i-l)*no_kmers + inx] * energy;
    }
    der_b[x[i]] -= cab * zb[i-l]*energy;

    for (int inx=0; inx<no_kmers; inx++)
    {
        zb_Ea_derivatives[i*no_kmers + inx] = der_a[inx];
        zb_Eb_derivatives[i*no_kmers + inx] = der_b[inx];
    }
 }



double inline cb_c(int d, double sf, double D, double sig)
{   
    if (d < 0)
        return 0;
    
    double eps = 1e-10;
    double diff = d - D; 
    double gaussian = exp(- (diff*diff) / (2.0 *(sig*sig + eps)));

    return gaussian*sf + 1 ;
}

double cb_D_derivative_c(int d, double sf, double D, double sig)
{    
    if (d < 0)
        return 0;

    double eps = 1e-10;
    double diff = d - D; 
    double der = sf*diff*exp(-diff*diff/(2.0 *(sig*sig + eps)))/(sig*sig+eps);
    return der;
    
}

double cb_sig_derivative_c(int d, double sf, double D, double sig)
{    
    if (d < 0)
        return 0;

    double eps = 1e-10;
    double diff = d - D; 
    double der = diff*diff*sig/((sig*sig + eps)*(sig*sig + eps));
    der *= sf*exp(-diff*diff/(2.0 *(sig*sig + eps)));
    return der;
    
}

double cb_sf_derivative_c(int d, double sf, double D, double sig)
{    
    if (d < 0)
        return 0;

    double eps = 1e-10;
    double diff = d - D; 
    double gaussian = exp(- (diff*diff) / (2.0 *(sig*sig + eps)));
    return gaussian;
    
}

