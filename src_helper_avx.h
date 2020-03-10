#include <math.h>
typedef double cfloat_t;

void assign_za_E_derivatives_c(long* , int , double* , double* , int , int , int,
                                 double* , double* , double* , double* ,
                                 double* , double* , double);
                                 
void assign_zb_E_derivatives_c(long* , int , double* , double* , int , int , int,
                                 double* , double* , double* , double* ,
                                 double* , double* , double, double, double, double);



double cb_c(int, double, double, double);
double cb_D_derivative_c(int, double, double, double);
double cb_sig_derivative_c(int, double, double, double);
double cb_sf_derivative_c(int, double, double, double);


