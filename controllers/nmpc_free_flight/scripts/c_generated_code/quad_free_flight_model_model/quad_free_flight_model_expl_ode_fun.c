/* This file was automatically generated by CasADi 3.6.7.
 *  It consists of: 
 *   1) content generated by CasADi runtime: not copyrighted
 *   2) template code copied from CasADi source: permissively licensed (MIT-0)
 *   3) user code: owned by the user
 *
 */
#ifdef __cplusplus
extern "C" {
#endif

/* How to prefix internal symbols */
#ifdef CASADI_CODEGEN_PREFIX
  #define CASADI_NAMESPACE_CONCAT(NS, ID) _CASADI_NAMESPACE_CONCAT(NS, ID)
  #define _CASADI_NAMESPACE_CONCAT(NS, ID) NS ## ID
  #define CASADI_PREFIX(ID) CASADI_NAMESPACE_CONCAT(CODEGEN_PREFIX, ID)
#else
  #define CASADI_PREFIX(ID) quad_free_flight_model_expl_ode_fun_ ## ID
#endif

#include <math.h>

#ifndef casadi_real
#define casadi_real double
#endif

#ifndef casadi_int
#define casadi_int int
#endif

/* Add prefix to internal symbols */
#define casadi_f0 CASADI_PREFIX(f0)
#define casadi_s0 CASADI_PREFIX(s0)
#define casadi_s1 CASADI_PREFIX(s1)
#define casadi_s2 CASADI_PREFIX(s2)
#define casadi_sq CASADI_PREFIX(sq)

/* Symbol visibility in DLLs */
#ifndef CASADI_SYMBOL_EXPORT
  #if defined(_WIN32) || defined(__WIN32__) || defined(__CYGWIN__)
    #if defined(STATIC_LINKED)
      #define CASADI_SYMBOL_EXPORT
    #else
      #define CASADI_SYMBOL_EXPORT __declspec(dllexport)
    #endif
  #elif defined(__GNUC__) && defined(GCC_HASCLASSVISIBILITY)
    #define CASADI_SYMBOL_EXPORT __attribute__ ((visibility ("default")))
  #else
    #define CASADI_SYMBOL_EXPORT
  #endif
#endif

casadi_real casadi_sq(casadi_real x) { return x*x;}

static const casadi_int casadi_s0[17] = {13, 1, 0, 13, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
static const casadi_int casadi_s1[8] = {4, 1, 0, 4, 0, 1, 2, 3};
static const casadi_int casadi_s2[4] = {0, 1, 0, 0};

/* quad_free_flight_model_expl_ode_fun:(i0[13],i1[4],i2[0])->(o0[13]) */
static int casadi_f0(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem) {
  casadi_real w0, w1, w2, w3, w4, w5, w6, w7, w8, w9, w10, w11, w12, w13, w14, w15;
  /* #0: @0 = input[0][7] */
  w0 = arg[0] ? arg[0][7] : 0;
  /* #1: output[0][0] = @0 */
  if (res[0]) res[0][0] = w0;
  /* #2: @0 = input[0][8] */
  w0 = arg[0] ? arg[0][8] : 0;
  /* #3: output[0][1] = @0 */
  if (res[0]) res[0][1] = w0;
  /* #4: @0 = input[0][9] */
  w0 = arg[0] ? arg[0][9] : 0;
  /* #5: output[0][2] = @0 */
  if (res[0]) res[0][2] = w0;
  /* #6: @0 = 0.5 */
  w0 = 5.0000000000000000e-01;
  /* #7: @1 = input[0][10] */
  w1 = arg[0] ? arg[0][10] : 0;
  /* #8: @2 = input[0][4] */
  w2 = arg[0] ? arg[0][4] : 0;
  /* #9: @3 = (@1*@2) */
  w3  = (w1*w2);
  /* #10: @3 = (-@3) */
  w3 = (- w3 );
  /* #11: @4 = input[0][11] */
  w4 = arg[0] ? arg[0][11] : 0;
  /* #12: @5 = input[0][5] */
  w5 = arg[0] ? arg[0][5] : 0;
  /* #13: @6 = (@4*@5) */
  w6  = (w4*w5);
  /* #14: @3 = (@3-@6) */
  w3 -= w6;
  /* #15: @6 = input[0][12] */
  w6 = arg[0] ? arg[0][12] : 0;
  /* #16: @7 = input[0][6] */
  w7 = arg[0] ? arg[0][6] : 0;
  /* #17: @8 = (@6*@7) */
  w8  = (w6*w7);
  /* #18: @3 = (@3-@8) */
  w3 -= w8;
  /* #19: @3 = (@0*@3) */
  w3  = (w0*w3);
  /* #20: output[0][3] = @3 */
  if (res[0]) res[0][3] = w3;
  /* #21: @3 = input[0][3] */
  w3 = arg[0] ? arg[0][3] : 0;
  /* #22: @8 = (@1*@3) */
  w8  = (w1*w3);
  /* #23: @9 = (@6*@5) */
  w9  = (w6*w5);
  /* #24: @8 = (@8+@9) */
  w8 += w9;
  /* #25: @9 = (@4*@7) */
  w9  = (w4*w7);
  /* #26: @8 = (@8-@9) */
  w8 -= w9;
  /* #27: @8 = (@0*@8) */
  w8  = (w0*w8);
  /* #28: output[0][4] = @8 */
  if (res[0]) res[0][4] = w8;
  /* #29: @8 = (@4*@3) */
  w8  = (w4*w3);
  /* #30: @9 = (@6*@2) */
  w9  = (w6*w2);
  /* #31: @8 = (@8-@9) */
  w8 -= w9;
  /* #32: @9 = (@1*@7) */
  w9  = (w1*w7);
  /* #33: @8 = (@8+@9) */
  w8 += w9;
  /* #34: @8 = (@0*@8) */
  w8  = (w0*w8);
  /* #35: output[0][5] = @8 */
  if (res[0]) res[0][5] = w8;
  /* #36: @8 = (@6*@3) */
  w8  = (w6*w3);
  /* #37: @9 = (@4*@2) */
  w9  = (w4*w2);
  /* #38: @8 = (@8+@9) */
  w8 += w9;
  /* #39: @9 = (@1*@5) */
  w9  = (w1*w5);
  /* #40: @8 = (@8-@9) */
  w8 -= w9;
  /* #41: @0 = (@0*@8) */
  w0 *= w8;
  /* #42: output[0][6] = @0 */
  if (res[0]) res[0][6] = w0;
  /* #43: @0 = input[1][0] */
  w0 = arg[1] ? arg[1][0] : 0;
  /* #44: @8 = input[1][1] */
  w8 = arg[1] ? arg[1][1] : 0;
  /* #45: @9 = (@0+@8) */
  w9  = (w0+w8);
  /* #46: @10 = input[1][2] */
  w10 = arg[1] ? arg[1][2] : 0;
  /* #47: @9 = (@9+@10) */
  w9 += w10;
  /* #48: @11 = input[1][3] */
  w11 = arg[1] ? arg[1][3] : 0;
  /* #49: @9 = (@9+@11) */
  w9 += w11;
  /* #50: @12 = (2.*@9) */
  w12 = (2.* w9 );
  /* #51: @13 = 2.5 */
  w13 = 2.5000000000000000e+00;
  /* #52: @12 = (@12/@13) */
  w12 /= w13;
  /* #53: @14 = (@5*@3) */
  w14  = (w5*w3);
  /* #54: @15 = (@7*@2) */
  w15  = (w7*w2);
  /* #55: @14 = (@14+@15) */
  w14 += w15;
  /* #56: @14 = (@12*@14) */
  w14  = (w12*w14);
  /* #57: output[0][7] = @14 */
  if (res[0]) res[0][7] = w14;
  /* #58: @14 = (@5*@7) */
  w14  = (w5*w7);
  /* #59: @15 = (@3*@2) */
  w15  = (w3*w2);
  /* #60: @14 = (@14-@15) */
  w14 -= w15;
  /* #61: @12 = (@12*@14) */
  w12 *= w14;
  /* #62: output[0][8] = @12 */
  if (res[0]) res[0][8] = w12;
  /* #63: @9 = (@9/@13) */
  w9 /= w13;
  /* #64: @3 = sq(@3) */
  w3 = casadi_sq( w3 );
  /* #65: @2 = sq(@2) */
  w2 = casadi_sq( w2 );
  /* #66: @3 = (@3-@2) */
  w3 -= w2;
  /* #67: @5 = sq(@5) */
  w5 = casadi_sq( w5 );
  /* #68: @3 = (@3-@5) */
  w3 -= w5;
  /* #69: @7 = sq(@7) */
  w7 = casadi_sq( w7 );
  /* #70: @3 = (@3+@7) */
  w3 += w7;
  /* #71: @9 = (@9*@3) */
  w9 *= w3;
  /* #72: @3 = 9.81 */
  w3 = 9.8100000000000005e+00;
  /* #73: @9 = (@9-@3) */
  w9 -= w3;
  /* #74: output[0][9] = @9 */
  if (res[0]) res[0][9] = w9;
  /* #75: @9 = 0.434783 */
  w9 = 4.3478260869565222e-01;
  /* #76: @3 = (-@0) */
  w3 = (- w0 );
  /* #77: @3 = (@3-@8) */
  w3 -= w8;
  /* #78: @3 = (@3+@10) */
  w3 += w10;
  /* #79: @3 = (@3+@11) */
  w3 += w11;
  /* #80: @7 = 4.5 */
  w7 = 4.5000000000000000e+00;
  /* #81: @5 = (@7*@4) */
  w5  = (w7*w4);
  /* #82: @5 = (@5*@6) */
  w5 *= w6;
  /* #83: @3 = (@3-@5) */
  w3 -= w5;
  /* #84: @5 = 2.3 */
  w5 = 2.2999999999999998e+00;
  /* #85: @2 = (@5*@6) */
  w2  = (w5*w6);
  /* #86: @13 = (@2*@4) */
  w13  = (w2*w4);
  /* #87: @3 = (@3+@13) */
  w3 += w13;
  /* #88: @3 = (@9*@3) */
  w3  = (w9*w3);
  /* #89: output[0][10] = @3 */
  if (res[0]) res[0][10] = w3;
  /* #90: @3 = (@0-@8) */
  w3  = (w0-w8);
  /* #91: @3 = (@3-@10) */
  w3 -= w10;
  /* #92: @3 = (@3+@11) */
  w3 += w11;
  /* #93: @2 = (@2*@1) */
  w2 *= w1;
  /* #94: @3 = (@3-@2) */
  w3 -= w2;
  /* #95: @7 = (@7*@1) */
  w7 *= w1;
  /* #96: @7 = (@7*@6) */
  w7 *= w6;
  /* #97: @3 = (@3+@7) */
  w3 += w7;
  /* #98: @9 = (@9*@3) */
  w9 *= w3;
  /* #99: output[0][11] = @9 */
  if (res[0]) res[0][11] = w9;
  /* #100: @9 = 0.222222 */
  w9 = 2.2222222222222221e-01;
  /* #101: @3 = -2 */
  w3 = -2.;
  /* #102: @3 = (@3*@0) */
  w3 *= w0;
  /* #103: @8 = (2.*@8) */
  w8 = (2.* w8 );
  /* #104: @3 = (@3+@8) */
  w3 += w8;
  /* #105: @10 = (2.*@10) */
  w10 = (2.* w10 );
  /* #106: @3 = (@3-@10) */
  w3 -= w10;
  /* #107: @11 = (2.*@11) */
  w11 = (2.* w11 );
  /* #108: @3 = (@3+@11) */
  w3 += w11;
  /* #109: @11 = (@5*@1) */
  w11  = (w5*w1);
  /* #110: @11 = (@11*@4) */
  w11 *= w4;
  /* #111: @3 = (@3-@11) */
  w3 -= w11;
  /* #112: @5 = (@5*@4) */
  w5 *= w4;
  /* #113: @5 = (@5*@1) */
  w5 *= w1;
  /* #114: @3 = (@3+@5) */
  w3 += w5;
  /* #115: @9 = (@9*@3) */
  w9 *= w3;
  /* #116: output[0][12] = @9 */
  if (res[0]) res[0][12] = w9;
  return 0;
}

CASADI_SYMBOL_EXPORT int quad_free_flight_model_expl_ode_fun(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem){
  return casadi_f0(arg, res, iw, w, mem);
}

CASADI_SYMBOL_EXPORT int quad_free_flight_model_expl_ode_fun_alloc_mem(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT int quad_free_flight_model_expl_ode_fun_init_mem(int mem) {
  return 0;
}

CASADI_SYMBOL_EXPORT void quad_free_flight_model_expl_ode_fun_free_mem(int mem) {
}

CASADI_SYMBOL_EXPORT int quad_free_flight_model_expl_ode_fun_checkout(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT void quad_free_flight_model_expl_ode_fun_release(int mem) {
}

CASADI_SYMBOL_EXPORT void quad_free_flight_model_expl_ode_fun_incref(void) {
}

CASADI_SYMBOL_EXPORT void quad_free_flight_model_expl_ode_fun_decref(void) {
}

CASADI_SYMBOL_EXPORT casadi_int quad_free_flight_model_expl_ode_fun_n_in(void) { return 3;}

CASADI_SYMBOL_EXPORT casadi_int quad_free_flight_model_expl_ode_fun_n_out(void) { return 1;}

CASADI_SYMBOL_EXPORT casadi_real quad_free_flight_model_expl_ode_fun_default_in(casadi_int i) {
  switch (i) {
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* quad_free_flight_model_expl_ode_fun_name_in(casadi_int i) {
  switch (i) {
    case 0: return "i0";
    case 1: return "i1";
    case 2: return "i2";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* quad_free_flight_model_expl_ode_fun_name_out(casadi_int i) {
  switch (i) {
    case 0: return "o0";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* quad_free_flight_model_expl_ode_fun_sparsity_in(casadi_int i) {
  switch (i) {
    case 0: return casadi_s0;
    case 1: return casadi_s1;
    case 2: return casadi_s2;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* quad_free_flight_model_expl_ode_fun_sparsity_out(casadi_int i) {
  switch (i) {
    case 0: return casadi_s0;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT int quad_free_flight_model_expl_ode_fun_work(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w) {
  if (sz_arg) *sz_arg = 5;
  if (sz_res) *sz_res = 2;
  if (sz_iw) *sz_iw = 0;
  if (sz_w) *sz_w = 16;
  return 0;
}

CASADI_SYMBOL_EXPORT int quad_free_flight_model_expl_ode_fun_work_bytes(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w) {
  if (sz_arg) *sz_arg = 5*sizeof(const casadi_real*);
  if (sz_res) *sz_res = 2*sizeof(casadi_real*);
  if (sz_iw) *sz_iw = 0*sizeof(casadi_int);
  if (sz_w) *sz_w = 16*sizeof(casadi_real);
  return 0;
}


#ifdef __cplusplus
} /* extern "C" */
#endif