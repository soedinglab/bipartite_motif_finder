#include <math.h>
typedef double c_float_t;

typedef struct DerParams {

  c_float_t* za_Ea_derivatives; 
  c_float_t* zb_Ea_derivatives; 
  c_float_t* za_Eb_derivatives; 
  c_float_t* zb_Eb_derivatives;
  c_float_t* der_a; 
  c_float_t* der_b;

} DerParams;

void initialize_DerParams(DerParams* params, int L, int no_kmers); 
void deinitialize_DerParams(DerParams* params);

void assign_za_E_derivatives_c(long* x, int i, double* za, double* zb, int L, int l, int no_kmers,
                                 DerParams* params, double* Ea, double* Eb, double cab);
                                 
void assign_zb_E_derivatives_c(long* x, int i, double* za, double* zb, int L, int l, int no_kmers,
                                 DerParams* params, double* Ea, double* Eb, double cab, double sf, double D, double sig);



double cb_c(int, double, double, double);
double cb_D_derivative_c(int, double, double, double);
double cb_sig_derivative_c(int, double, double, double);
double cb_sf_derivative_c(int, double, double, double);


