#ifndef FANG_CPU_GEMM_H
#define FANG_CPU_GEMM_H

#include <platform/memory.h>
#include <compiler.h>
#include <stdbool.h>

#if defined(FANG_USE_AVX512) || defined(FANG_USE_AVX2)
#include <immintrin.h>
#endif  // FANG_USE_AVX512 or FANG_USE_AVX2

/* ================ HELPER MACROS ================ */

/* Fang use row-major order exclusively. */
#define _alpha(i, j)    x[i * ld_x + j]
#define _beta(i, j)     y[i * ld_y + j]
#define _gamma(i, j)    dest[i * ld_dest + j]

#ifndef _FANG_MIN
#define _FANG_MIN(x, y)    ((x) < (y) ? (x) : (y))
#endif  // _FANG_MIN

/* ================ HELPER MACROS END ================ */


/* ================ DATA TYPES ================ */

/* Single-precision GEMM micro-kernel. */
typedef void (*_fang_sgemm_ukernel_t)(int k, float beta, float *restrict dest,
    int ld_dest, float alpha, float *restrict x, float *restrict y);

/* ================ DATA TYPES END ================ */


/* ================ DATA STRUCTURES ================ */

/* SGEMM micro-kernels. */
extern _fang_sgemm_ukernel_t _sgemm_ukernels[];

/* SGEMM micro-kernel strides, MRxNR:
 *     6x16: 0
 */
extern int _sgemm_mr[];
extern int _sgemm_nr[];

/* ================ DATA STRUCTURES END ================ */


/* ================ DECLARATIONS ================ */

/* Single-precision (float32) GEMM. */
FANG_HOT int _fang_sgemm(bool transp_x, bool transp_y, int m, int n, int k,
    float beta, float *restrict dest, int ld_dest, float alpha,
    float *restrict x, int ld_x, float *restrict y, int ld_y);

/* SGEMM packing subroutine. */
/* NOTE: This packing subroutine favors the row-major access pattern of matrices
 *   while packing (say, when packing for matrix `y`). Hence, to use this
 *   subroutine for packing matrix `x`, leverage the `transpose` parameter
 *   because due to rank-1 update computation in the micro-kernel, the `x`
 *   matrix access pattern is reversed.
 */
FANG_HOT void _fang_sgemm_pack(int k, int xn, int stride, float *restrict x,
    int ld_x, float *restrict x_tilde, bool transpose);

/* ================ DECLARATIONS END ================ */


/* ================ INLINE FIVE-LOOPS DEFINITIONS ================ */

/* Fang use slightly modified version of GEMM in BLIS
 * (https://github.com/flame/blis), favoring row-major order exclusively. This
 * is also slightly modified version of GotoBLAS GEMM algorithm having 5 outer
 * loops around the micro-kernel with opposed to 3 outer loops.
 */

#define _FANG_OUTER_LOOPS(dtype, gemm, gemmu)                                   \
                                                                                \
/* Loop 1, slices matrix `dest` and KCxNC panel of `y` into KCxNR
   micro-panels and stream from KCxNC block of `y` from L2 cache. */            \
FANG_HOT FANG_INLINE FANG_FLATTEN static inline void                            \
_fang_##gemm##_loop1(int m, int n, int k,                                       \
    dtype beta,                                                                 \
    dtype *restrict dest, int ld_dest,                                          \
    dtype alpha,                                                                \
    dtype *restrict x_packed,                                                   \
    dtype *restrict y_packed)                                                   \
{                                                                               \
    int stride_mr = _##gemm##_mr[FANG_##gemmu##_KERNEL];                        \
    int stride_nr = _##gemm##_nr[FANG_##gemmu##_KERNEL];                        \
                                                                                \
    /* TODO: Parallelize loop 3. */                                             \
    for(int j = 0; j < n; j += stride_nr) {                                     \
        int jb = _FANG_MIN(stride_nr, n - j);                                   \
                                                                                \
        if(FANG_LIKELY(m == stride_mr && jb == stride_nr))                      \
            /* Call micro-kernel. */                                            \
            _##gemm##_ukernels[FANG_##gemmu##_KERNEL](k, beta,                  \
                &_gamma(0, j), ld_dest, alpha, x_packed, &y_packed[k * j]);     \
        else {                                                                  \
            dtype dest_shell[stride_mr][stride_nr] FANG_ALIGNAS(64);            \
                                                                                \
            /* Copy original C. */                                              \
            if(FANG_UNLIKELY(beta != (dtype) 0)) {                              \
                for(int ir = 0; ir < m; ir++) {                                 \
                    for(int jr = 0; jr < jb; jr++)                              \
                        dest_shell[ir][jr] = _gamma(ir, j + jr);                \
                }                                                               \
            }                                                                   \
                                                                                \
            /* Call micro-kernel with the shell over `dest` matrix. */          \
            _##gemm##_ukernels[FANG_##gemmu##_KERNEL](k, beta, (dtype *)        \
                dest_shell, stride_nr, alpha, x_packed, &y_packed[k * j]);      \
                                                                                \
            /* Copy the shell data to original `dest` matrix. */                \
            for(int ir = 0; ir < m; ir++) {                                     \
                for(int jr = 0; jr < jb; jr++)                                  \
                    _gamma(ir, j + jr) = dest_shell[ir][jr];                    \
            }                                                                   \
        }                                                                       \
    }                                                                           \
}                                                                               \
                                                                                \
/* Loop 2, slices matrix `dest` and MCxKC panel of `x` into MRxKC
   micro-panels and keeps the micro-panels in L1 cache. */                      \
FANG_HOT FANG_INLINE FANG_FLATTEN static inline void                            \
_fang_##gemm##_loop2(int m, int n, int k,                                       \
    dtype beta,                                                                 \
    dtype *restrict dest, int ld_dest,                                          \
    dtype alpha,                                                                \
    dtype *restrict x_packed,                                                   \
    dtype *restrict y_packed)                                                   \
{                                                                               \
    int stride_mr = _##gemm##_mr[FANG_##gemmu##_KERNEL];                        \
                                                                                \
    /* TODO: Parallelize loop 3. */                                             \
    for(int i = 0; i < m; i += stride_mr) {                                     \
        int ib = _FANG_MIN(stride_mr, m - i);                                   \
                                                                                \
        /* Dispatch to loop 1. */                                               \
        _fang_##gemm##_loop1(ib, n, k, beta, &_gamma(i, 0), ld_dest,            \
            alpha, &x_packed[i * k], y_packed);                                 \
    }                                                                           \
}                                                                               \
                                                                                \
/* Loop 3, slices matrix `dest` and `y` in terms of column cache block (KC).
   This loop ensures KCxNC block from `y` stays in the L2 cache. */             \
FANG_HOT FANG_INLINE FANG_FLATTEN static inline void                            \
_fang_##gemm##_loop3(bool transp_y,                                             \
    int m, int n, int k,                                                        \
    dtype beta,                                                                 \
    dtype *restrict dest, int ld_dest,                                          \
    dtype alpha,                                                                \
    dtype *restrict x_packed,                                                   \
    dtype *restrict y, int ld_y,                                                \
    dtype *restrict y_tilde)                                                    \
{                                                                               \
    /* TODO: Parallelize loop 3. */                                             \
    for(int j = 0; j < n; j += FANG_##gemmu##_NC) {                             \
        int jb = _FANG_MIN(FANG_##gemmu##_NC, n - j);                           \
                                                                                \
        /* Pack KCxNC block of matrix `y` and keep in L2 cache. */              \
        _fang_##gemm##_pack(k, jb, _##gemm##_nr[FANG_##gemmu##_KERNEL],         \
            &_beta(0, j), ld_y, y_tilde, transp_y);                             \
                                                                                \
        /* Prefetch to keep `x` in L3 cache. */                                 \
        FANG_PREFETCH(y_tilde, FANG_PREFETCH_READ,                              \
            FANG_PREFETCH_LOCALITY_D2);                                         \
                                                                                \
        /* Dispatch to loop 2. */                                               \
        _fang_##gemm##_loop2(m, jb, k, beta, &_gamma(0, j), ld_dest, alpha,     \
            x_packed, y_tilde);                                                 \
    }                                                                           \
}                                                                               \
                                                                                \
/* Loop 4, slices matrix `x` and `y` in terms of common dimension cache
   block (KC). This loop ensures MCxKC panel from `x` stays in the L3
   cache. */                                                                    \
FANG_HOT FANG_INLINE FANG_FLATTEN static inline void                            \
_fang_##gemm##_loop4(bool transp_x, bool transp_y,                              \
    int m, int n, int k,                                                        \
    dtype beta,                                                                 \
    dtype *restrict dest, int ld_dest,                                          \
    dtype alpha,                                                                \
    dtype *restrict x, int ld_x,                                                \
    dtype *restrict y, int ld_y,                                                \
    dtype *restrict x_tilde,                                                    \
    dtype *restrict y_tilde)                                                    \
{                                                                               \
    for(int p = 0; p < k; p += FANG_##gemmu##_KC) {                             \
        int pb = _FANG_MIN(FANG_##gemmu##_KC, k - p);                           \
        /* Beta needs to be applied only once. */                               \
        dtype _bet = (p == 0) ? beta : (dtype) 1;                               \
        /* Pack MCxKC panel of matrix `x` and keep in L3 cache. */              \
        _fang_##gemm##_pack(pb, m, _##gemm##_mr[FANG_##gemmu##_KERNEL],         \
            &_alpha(0, p), ld_x, x_tilde, !transp_x);                           \
                                                                                \
        /* Prefetch to keep `x` in L3 cache. */                                 \
        FANG_PREFETCH(x_tilde, FANG_PREFETCH_READ,                              \
            FANG_PREFETCH_LOCALITY_D1);                                         \
                                                                                \
        /* Dispatch to loop 3. */                                               \
        _fang_##gemm##_loop3(transp_y, m, n, pb, _bet, dest, ld_dest, alpha,    \
            x_tilde, &_beta(p, 0), ld_y, y_tilde);                              \
    }                                                                           \
}                                                                               \
                                                                                \
/* Loop 5, slices matrix `dest` and `x` in terms of row cache block (MC). */    \
FANG_HOT FANG_INLINE FANG_FLATTEN static inline void                            \
_fang_##gemm##_loop5(bool transp_x, bool transp_y,                              \
    int m, int n, int k,                                                        \
    dtype beta,                                                                 \
    dtype *restrict dest, int ld_dest,                                          \
    dtype alpha,                                                                \
    dtype *restrict x, int ld_x,                                                \
    dtype *restrict y, int ld_y)                                                \
{                                                                               \
    /* Memory for packed MCxKC panel of `x` and KCxNC block of `y`. */          \
    dtype *restrict x_tilde = _fang_aligned_malloc(FANG_##gemmu##_MC *          \
        FANG_##gemmu##_KC * sizeof(dtype), 64);                                 \
    dtype *restrict y_tilde = _fang_aligned_malloc(FANG_##gemmu##_KC *          \
        FANG_##gemmu##_NC * sizeof(dtype), 64);                                 \
                                                                                \
    /* TODO: Parallelize loop 5. */                                             \
    for(int i = 0; i < m; i += FANG_##gemmu##_MC) {                             \
        int ib = _FANG_MIN(FANG_##gemmu##_MC, m - i);                           \
                                                                                \
        /* Dispatch to loop 4. */                                               \
        _fang_##gemm##_loop4(transp_x, transp_y, ib, n, k, beta,                \
            &_gamma(i, 0), ld_dest, alpha, &_alpha(i, 0), ld_x, y, ld_y,        \
            x_tilde, y_tilde);                                                  \
    }                                                                           \
                                                                                \
    free(x_tilde);                                                              \
    free(y_tilde);                                                              \
}                                                                               \

/* ================ INLINE FIVE-LOOPS DEFINITIONS END ================ */

#endif  // FANG_CPU_GEMM_H
