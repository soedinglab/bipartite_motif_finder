#ifndef SRC_HELPER
#define SRC_HELPER

#include <math.h>
#include <assert.h>

//typedef float c_float_t;
typedef float c_float_t;

typedef struct DerParams {

  c_float_t* za_Ea_derivatives; 
  c_float_t* zb_Ea_derivatives; 
  c_float_t* za_Eb_derivatives; 
  c_float_t* zb_Eb_derivatives;
  c_float_t* der_a; 
  c_float_t* der_b;

} DerParams;

static void initialize_array(c_float_t* arr, c_float_t value, int length);
c_float_t sum_array_c(c_float_t* arr, int length);
void sum_mat_rows(c_float_t* out, c_float_t* mat, int n_row, int n_col);

void initialize_DerParams(DerParams* params, int L, int no_kmers); 
void deinitialize_DerParams(DerParams* params);

void assign_za_c(int i, c_float_t* za, c_float_t* zb, c_float_t concentration_times_energy, int l);
void assign_zb_c(long* x, int i, c_float_t* za, c_float_t* zb, c_float_t* Eb, c_float_t cab, c_float_t sf, c_float_t r, c_float_t p, int l);

void assign_za_E_derivatives_c(long* x, int i, c_float_t* za, c_float_t* zb, int L, int l, int no_kmers,
                                 DerParams* params, c_float_t* Ea, c_float_t* Eb, c_float_t cab);
                                 
void assign_zb_E_derivatives_c(long* x, int i, c_float_t* za, c_float_t* zb, int L, int l, int no_kmers,
                                 DerParams* params, c_float_t* Ea, c_float_t* Eb, c_float_t cab, c_float_t sf, c_float_t r, c_float_t p);

void assign_za_r_derivative_c(int i, c_float_t* za_r_derivatives, c_float_t* zb_r_derivatives, c_float_t concentration_times_energy, int l);
void assign_za_p_derivative_c(int i, c_float_t* za_p_derivatives, c_float_t* zb_p_derivatives, c_float_t concentration_times_energy, int l);
void assign_za_sf_derivative_c(int i, c_float_t* za_sf_derivatives, c_float_t* zb_sf_derivatives, c_float_t concentration_times_energy, int l);

void assign_zb_r_derivative_c(int i, c_float_t* za, c_float_t* za_r_derivatives, c_float_t* zb_r_derivatives, c_float_t energy_b, 
                                     c_float_t cab, c_float_t sf,c_float_t r, c_float_t p, int l);
void assign_zb_p_derivative_c(int i, c_float_t* za, c_float_t* za_p_derivatives, c_float_t* zb_p_derivatives, c_float_t energy_b, 
                                     c_float_t cab, c_float_t sf, c_float_t r, c_float_t p, int l);
void assign_zb_sf_derivative_c(int i, c_float_t* za, c_float_t* za_sf_derivatives, c_float_t* zb_sf_derivatives, c_float_t energy_b, 
                                     c_float_t cab, c_float_t sf, c_float_t r, c_float_t p, int l);

c_float_t cb_c(int, c_float_t, c_float_t, c_float_t);
c_float_t cb_r_derivative_c(int, c_float_t, c_float_t, c_float_t);
c_float_t cb_p_derivative_c(int, c_float_t, c_float_t, c_float_t);
c_float_t cb_sf_derivative_c(int, c_float_t, c_float_t, c_float_t);
c_float_t digamma(c_float_t);

#endif
