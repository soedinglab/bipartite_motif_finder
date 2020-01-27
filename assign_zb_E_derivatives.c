#include "assign_zb_E_derivatives.h"


void assign_zb_E_derivatives_c(long* x, int i, int inx, double* za, double* zb, int L, int l, double l_p,
                                 double* za_Ea_derivatives, double* zb_Ea_derivatives, double* za_Eb_derivatives, double* zb_Eb_derivatives,
                                 double* Ea, double* Eb, double cab, double sf, double D, double sig)
                                 {
    int identical = (inx == x[i]);
    
    double der_b = zb_Eb_derivatives[inx*L+i-1];
    double der_a = zb_Ea_derivatives[inx*L+i-1];

    double energy = exp(-Eb[x[i]]);
    int j;
    
    if (i == l-1)  {
        der_b += -cab*identical*energy;
        der_a += 0;
    }
        
    else  {
        for (j=0; j<i-l+1; ++j)
        {
            der_b += cb_c(i-j-l, sf, D, sig) * ((za_Eb_derivatives[inx*L+j]*energy - za[j]*energy*identical));
            der_a += cb_c(i-j-l, sf, D, sig) * za_Ea_derivatives[inx*L+j]*energy;  
        }
        der_b += cab * (zb_Eb_derivatives[inx*L+i-l]*energy - zb[i-l]*energy*identical); 
        der_a += cab * zb_Ea_derivatives[inx*L+i-l]*energy;
    }
    
    zb_Ea_derivatives[inx*L+i] = der_a;
    zb_Eb_derivatives[inx*L+i] = der_b;
    
    return;

 }


double inline cb_c(int d, double sf, double D, double sig)
{   
    if (d < 0)
        return 0;
    
    double diff = d + 1 - D; 
    double gaussian = exp(- (diff*diff) / (2.0 *sig*sig)) / (sig*sqrt(2*M_PI));
    return gaussian*sf + 1 ;
}

double cb_D_derivative_c(int d, double sf, double D, double sig)
{    
    if (d < 0)
        return 0;

    double diff = d + 1 - D; 
    double der = sf*diff*exp(-diff*diff/(2*sig*sig))/(sig*sig*sig*sqrt(2*M_PI));
    return der;
    
}

double cb_sig_derivative_c(int d, double sf, double D, double sig)
{    
    if (d < 0)
        return 0;

    double diff = d + 1 - D; 
    double sig2 = sig*sig;
    double der = diff*diff/(sig2*sig2) - 1/sig2;
    der *= (sf*exp(-diff*diff/(2*sig2)))/sqrt(2*M_PI);
    return der;
    
}

double cb_sf_derivative_c(int d, double sf, double D, double sig)
{    
    if (d < 0)
        return 0;

    double diff = d + 1 - D; 
    double der = sf*exp(-diff*diff/(2*sig*sig))/(sig*sqrt(2*M_PI));
    return der;
    
}

