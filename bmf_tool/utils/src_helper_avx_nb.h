#ifndef SRC_HELPER
#define SRC_HELPER

#include <math.h>
#include <assert.h>

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
void assign_zb_c(long* x, int i, double* za, double* zb, double* Eb, double cab, double sf, double r, double p, int l);

void assign_za_E_derivatives_c(long* x, int i, double* za, double* zb, int L, int l, int no_kmers,
                                 DerParams* params, double* Ea, double* Eb, double cab);
                                 
void assign_zb_E_derivatives_c(long* x, int i, double* za, double* zb, int L, int l, int no_kmers,
                                 DerParams* params, double* Ea, double* Eb, double cab, double sf, double r, double p);

void assign_za_r_derivative_c(int i, double* za_r_derivatives, double* zb_r_derivatives, double concentration_times_energy, int l);
void assign_za_p_derivative_c(int i, double* za_p_derivatives, double* zb_p_derivatives, double concentration_times_energy, int l);
void assign_za_sf_derivative_c(int i, double* za_sf_derivatives, double* zb_sf_derivatives, double concentration_times_energy, int l);

void assign_zb_r_derivative_c(int i, double* za, double* za_r_derivatives, double* zb_r_derivatives, double energy_b, 
                                     double cab, double sf,double r, double p, int l);
void assign_zb_p_derivative_c(int i, double* za, double* za_p_derivatives, double* zb_p_derivatives, double energy_b, 
                                     double cab, double sf, double r, double p, int l);
void assign_zb_sf_derivative_c(int i, double* za, double* za_sf_derivatives, double* zb_sf_derivatives, double energy_b, 
                                     double cab, double sf, double r, double p, int l);

double cb_c(int, double, double, double);
double cb_r_derivative_c(int, double, double, double);
double cb_p_derivative_c(int, double, double, double);
double cb_sf_derivative_c(int, double, double, double);
double digamma(double);

#endif
