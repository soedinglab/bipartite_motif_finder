#include "assign_zb_E_derivatives.h"


void assign_zb_E_derivatives_c(long* x, int i, int inx, double* za, double* zb, int L, int l, double l_p,
                                 double* za_Ea_derivatives, double* zb_Ea_derivatives, double* za_Eb_derivatives, double* zb_Eb_derivatives,
                                 double* Ea, double* Eb, double cab, double D)
                                 {
    int identical = (inx == x[i]);
    
    double der_b = zb_Eb_derivatives[inx*L+i-1];
    double der_a = zb_Ea_derivatives[inx*L+i-1];
    int j;
    
    if (i == l-1)  {
        der_b += -cab*identical*exp(-Eb[x[i]]);
        der_a += 0;
    }
        
    else  {
        for (j=0; j<i-l+1; ++j)
        {
            der_b += cb_c(i-j-l, cab, D, l_p) * ((za_Eb_derivatives[inx*L+j]*exp(-Eb[x[i]]) - za[j]*exp(-Eb[x[i]])*identical));
            der_b += cab * (zb_Eb_derivatives[inx*L+j]*exp(-Eb[x[i]]) - zb[j]*exp(-Eb[x[i]])*identical);
            
            der_a += cb_c(i-j-l, cab, D, l_p) * za_Ea_derivatives[inx*L+j]*exp(-Eb[x[i]]);
            der_a += cab * zb_Ea_derivatives[inx*L+j]*exp(-Eb[x[i]]);
        } 
    }
    
    zb_Ea_derivatives[inx*L+i] = der_a;
    zb_Eb_derivatives[inx*L+i] = der_b;
    
    return;

 }


double cb_c(int d, double cab, double D, double l_p)
{   
    if (d < 0)
        return 0;
    double sig = 1 / (3*(d+1)*l_p);
    double gaussian = exp(-pow(D,2) / (2 * pow(sig, 2.))) / pow(2*M_PI*pow(sig,2),3/2);
    return cab + gaussian ;
}

