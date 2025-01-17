#include <env/cpu/gemm.h>
#include <fang/status.h>
#include <tune.h>

#include <string.h>


/* ================ PRIVATE HELPER MACROS ================ */

/* Access element from a matrix upanel. */
#define X(i, j)     x[(i) * ld_x + (j)]

/* ================ PRIVATE HELPER MACROS END ================ */


/* ================ DEFINITIONS ================ */

/* Instantiate the five outer loops. */
_FANG_OUTER_LOOPS(float, sgemm, SGEMM)

/* Single-precision (float32) GEMM. */
int _fang_sgemm(bool transp_x, bool transp_y, int m, int n, int k,
    float beta, float *restrict dest, int ld_dest, float alpha,
    float *restrict x, int ld_x, float *restrict y, int ld_y)
{
    int res = FANG_OK;

    /* When `alpha` is 0, scale `dest` and return. */
    if(FANG_UNLIKELY(alpha == 0)) {
        if(FANG_UNLIKELY(beta == 0)) {
            /* Zero out `dest`. */
            memset(dest, 0, m * n * sizeof(float));
            goto out;
        } else {
            /* Scale by beta. */
            for(int i = 0; i < m * n; i++)
                dest[i] *= beta;

            goto out;
        }
    }

    /* Dispatch to five outer loops. */
    _fang_sgemm_loop5(transp_x, transp_y, m, n, k, beta, dest, ld_dest, alpha,
        x, ld_x, y, ld_y);

out:
    return res;
}

/* SGEMM packing subroutine. */
/* NOTE: This packing subroutine favors the row-major access pattern of matrices
 *   while packing (say, when packing for matrix `y`). Hence, to use this
 *   subroutine for packing matrix `x`, leverage the `transpose` parameter
 *   because due to rank-1 update computation in the micro-kernel, the `x`
 *   matrix access pattern is reversed.
 */
void _fang_sgemm_pack(int k, int xn, int stride, float *restrict x, int ld_x,
    float *restrict x_tilde, bool transpose)
{
    /* `stride` is either `MR` or `NR`. In other words, `stride` is register
       blocking dimension size. */

    /* Try prefetching `x_tilde` and `x` beforehand. */
    FANG_PREFETCH(x_tilde, FANG_PREFETCH_WRITE, FANG_PREFETCH_LOCALITY_D3);
    FANG_PREFETCH(x, FANG_PREFETCH_READ, FANG_PREFETCH_LOCALITY_D3);

    /* Unroll loops upto factor of 8. */
#if defined(__GNUC__)
    #pragma GCC unroll 8
#elif defined(__clang__)
    #pragma clang loop unroll_count(8)
#endif  // unroll 8
    for(int ir = 0; ir < xn; ir += stride) {
        int irb = _FANG_MIN(stride, xn - ir);
        for(int p = 0; p < k; p++) {
            int i = 0;
            for(; i < irb; i++)
                *x_tilde++ = transpose ? X(ir + i, p) : X(p, ir + i);

            /* Fill remainders with 0. */
            for(; i < stride; i++)
                *x_tilde++ = 0.0f;
        }
    }
}

/* ================ DEFINITIONS END ================ */
