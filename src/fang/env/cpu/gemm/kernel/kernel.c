#include <env/cpu/gemm.h>

/* Defualt target cpu architecture is "haswell". */
#ifndef FANG_GEMM_CPU_TARGET
#define FANG_GEMM_CPU_TARGET    haswell
#endif   // FANG_GEMM_CPU_TARGET

/* ================ HELPER MACROS ================ */

/* Unpacks the value of a macro and stringifies it. */
#define _STRINGIFY(x)    #x
#define _UNPACK(x)       _STRINGIFY(x)

/* ================ HELPER MACROS END ================ */


/* Include SGEMM kernels. */
#include _UNPACK(FANG_GEMM_CPU_TARGET/fang_sgemm_6x16_ukernel_asm.c.inc)


/* ================ DATA STRUCTURES ================ */

/* SGEMM micro-kernels. */
_fang_sgemm_ukernel_t _sgemm_ukernels[] = {
    _fang_sgemm_6x16_ukernel
};

/* SGEMM micro-kernel strides, MRxNR:
 *     6x16: 0
 */
int _sgemm_mr[] = {  6 };
int _sgemm_nr[] = { 16 };

/* ================ DATA STRUCTURES END ================ */
