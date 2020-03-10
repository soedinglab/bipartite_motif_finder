#include "src_helper_avx.h"

void assign_za_E_derivatives_c(long* x, int i, double* za, double* zb, int L, int l, int no_kmers,
                                 double* za_Ea_derivatives, double* zb_Ea_derivatives, double* za_Eb_derivatives, double* zb_Eb_derivatives,
                                 double* Ea, double* Eb, double cab)
{  
    int inx;
    for (inx=0; inx<no_kmers; inx++)
    {  
        int identical = (inx == x[i]);
            
        double energy = exp(-Ea[x[i]])*cab;

        double der_a = zb_Ea_derivatives[(i-l)*no_kmers + inx] - zb[i-l]*identical; 
        double der_b = zb_Eb_derivatives[(i-l)*no_kmers + inx];

        int j;
        for (j=0; j<i-l+1; ++j)
        {
            der_a += za_Ea_derivatives[j*no_kmers + inx] - za[j]*identical;
            der_b += za_Eb_derivatives[j*no_kmers + inx];
        }

        za_Ea_derivatives[i*no_kmers + inx] = der_a*energy;
        za_Eb_derivatives[i*no_kmers + inx] = der_b*energy;
    }
}

void assign_zb_E_derivatives_c(long* x, int i, double* za, double* zb, int L, int l, int no_kmers,
                                 double* za_Ea_derivatives, double* zb_Ea_derivatives, double* za_Eb_derivatives, double* zb_Eb_derivatives,
                                 double* Ea, double* Eb, double cab, double sf, double D, double sig)
 {
    int inx;
    for (inx=0; inx<no_kmers; inx++)
    {
        int identical = (inx == x[i]);
        
        double der_b = zb_Eb_derivatives[(i-l)*no_kmers + inx];
        double der_a = zb_Ea_derivatives[(i-l)*no_kmers + inx];

        double energy = exp(-Eb[x[i]]);
        int j;
        
        for (j=0; j<i-l+1; ++j)
        {
            der_b += cb_c(i-j-l, sf, D, sig) * ((za_Eb_derivatives[j*no_kmers + inx]*energy - za[j]*energy*identical));
            der_a += cb_c(i-j-l, sf, D, sig) * za_Ea_derivatives[j*no_kmers + inx]*energy;  
        }
        der_b += cab * (zb_Eb_derivatives[(i-l)*no_kmers + inx]*energy - zb[i-l]*energy*identical); 
        der_a += cab * zb_Ea_derivatives[(i-l)*no_kmers + inx]*energy;
        
        zb_Ea_derivatives[i*no_kmers + inx] = der_a;
        zb_Eb_derivatives[i*no_kmers + inx] = der_b;
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

