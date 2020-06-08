#ifndef SRC_HELPER
#define SRC_HELPER

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

static void initialize_array(c_float_t* arr, c_float_t value, int length);

double sum_array_c(double* arr, int length);
void sum_mat_rows(double* out, double* mat, int n_row, int n_col);

void initialize_DerParams(DerParams* params, int L, int no_kmers); 
void deinitialize_DerParams(DerParams* params);

void assign_za_c(int i, double* za, double* zb, double concentration_times_energy, int l);
void assign_zb_c(long* x, int i, double* za, double* zb, double* Eb, double cab, double sf, double D, double sig, int l);

void assign_za_E_derivatives_c(long* x, int i, double* za, double* zb, int L, int l, int no_kmers,
                                 DerParams* params, double* Ea, double* Eb, double cab);
                                 
void assign_zb_E_derivatives_c(long* x, int i, double* za, double* zb, int L, int l, int no_kmers,
                                 DerParams* params, double* Ea, double* Eb, double cab, double sf, double D, double sig);

void assign_za_D_derivative_c(int i, double* za_D_derivatives, double* zb_D_derivatives, double concentration_times_energy, int l);
void assign_za_sig_derivative_c(int i, double* za_sig_derivatives, double* zb_sig_derivatives, double concentration_times_energy, int l);
void assign_za_sf_derivative_c(int i, double* za_sf_derivatives, double* zb_sf_derivatives, double concentration_times_energy, int l);

void assign_zb_D_derivative_c(int i, double* za, double* za_D_derivatives, double* zb_D_derivatives, double energy_b, 
                                     double cab, double sf, double D , double sig, int l);
void assign_zb_sig_derivative_c(int i, double* za, double* za_sig_derivatives, double* zb_sig_derivatives, double energy_b, 
                                     double cab, double sf, double D , double sig, int l);
void assign_zb_sf_derivative_c(int i, double* za, double* za_sf_derivatives, double* zb_sf_derivatives, double energy_b, 
                                     double cab, double sf, double D , double sig, int l);

double cb_c(int, double, double, double);
double cb_D_derivative_c(int, double, double, double);
double cb_sig_derivative_c(int, double, double, double);
double cb_sf_derivative_c(int, double, double, double);


#endif
