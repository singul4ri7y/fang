#ifndef FANG_TENSOR_H
#define FANG_TENSOR_H

#include <fang/config.h>
#include <fang/type.h>
#include <compiler.h>
#include <stdio.h>

/* ================ HELPER MACROS ================ */

/* ======== DIMENSION HELPER MACROS ======== */

/* Fill in and pass the dimension tensor. */
#define FANG_DIM(...)    (fang_ten_dim_t) {                              \
    .dims = (uint32_t []) { __VA_ARGS__ },                               \
    .ndims = sizeof((uint32_t []) { __VA_ARGS__ }) / sizeof(uint32_t)    \
}

/* Alias. */
#define $D(...)          FANG_DIM(__VA_ARGS__)

/* ======== DIMENSION HELPER MACROS END ======== */

/* ======== FUNCTION WRAPPER MACROS ======== */

/* Print tensor to stdout. */
#define FANG_TEN_PRINT(ten)        fang_ten_fprint(ten, #ten, 0, stdout)

/* Print tensor to some other file. */
#define FANG_TEN_FPRINT(ten, f)    fang_ten_fprint(ten, #ten, 0, f)

/* ======== FUNCTION WRAPPER MACROS END ======== */

/* ======== TENSOR BROADCASTING MACROS ======== */

/* Macros to dictate broadcast patterns, used to provide hints to lower-level
 * on how broadcasting should play out a.k.a fast-routes. There are 6
 * broadcast patterns:
 *   1. Boradcast scalar against N-dimensional tensor. Denoted with 1.
 *   2. Broadcast row-major vector against N-dimensional tensor. Denoted with 2.
 *   3. Broadcast col-major vector against N-dimensional tensor. Denoted with 3.
 *   4. Broadcast a matrix against N-dimentional tensor (Perceived as array of
 *      matrices of same shape if flattened at some degree). Denoted with 4.
 *   5. Broadcast dimension is unknown. Denoted with 5.
 *   6. No broadcasting :). Denoted with 0.
 */

/* NOTE: Here, dimensions e.g. (1, 1, 1, 1, 7) is considered a row vector with
 *   only single dimension, just like (7) (or (1, 7) if considered in context of
 *   matrices). Similarly, column vector and matrix broadcasting follows the
 *   same dimension contraction if leading dimensions are 1.
 */

#define FANG_NO_BCAST             0
#define FANG_BCAST_SCALAR         1
#define FANG_BCAST_ROWVEC         2
#define FANG_BCAST_COLVEC         3
#define FANG_BCAST_MATRIX         4
#define FANG_BCAST_UNKNOWN        5

/* ======== TENSOR BROADCASTING MACROS END ======== */

/* ================ HELPER MACROS END ================ */


/* ================ DATA STRUCTURES ================ */

/* Type of each tensor. */
typedef enum fang_ten_type {
    FANG_TEN_TYPE_DENSE,
    FANG_TEN_TYPE_SPARSE
} fang_ten_type_t;

/* Sparse tensor data representation using COO encoding. */
typedef struct fang_ten_sparse_coo {
    /* Number of non-zero elements. */
    int nnz;

    /* Indicies for non-zero elements. */
    uint32_t **idx;

    /* Contiguous non-zero elements. Row-major order. */
    void *data;
} fang_ten_sparse_coo_t;

/* Data type of each tensor. */
/* NOTE: TRY NOT TO CHANGE THE ORDERING. */
typedef enum fang_ten_dtype {
    FANG_TEN_DTYPE_INT8,
    FANG_TEN_DTYPE_INT16,
    FANG_TEN_DTYPE_INT32,
    FANG_TEN_DTYPE_INT64,
    FANG_TEN_DTYPE_UINT8,
    FANG_TEN_DTYPE_UINT16,
    FANG_TEN_DTYPE_UINT32,
    FANG_TEN_DTYPE_UINT64,
    FANG_TEN_DTYPE_FLOAT8,
    FANG_TEN_DTYPE_FLOAT16,
    FANG_TEN_DTYPE_BFLOAT16,
    FANG_TEN_DTYPE_FLOAT32,
    FANG_TEN_DTYPE_FLOAT64,

    FANG_TEN_DTYPE_INVALID = -1
} fang_ten_dtype_t;

/* Represents a single tensor. */
typedef struct fang_ten {
    /* Environment ID with which the tensor is created. */
    uint16_t eid;

    /* The number of dimensions. */
    uint16_t ndims;

    /* Type of tensor. */
    fang_ten_type_t typ;

    /* Type of data tensor is holding. */
    fang_ten_dtype_t dtyp;

    /* Tensor's dimension. */
    uint32_t *dims;

    /* Respective strides for each dimension. */
    uint32_t *strides;

    /* Tensor data. Row-major order. */
    union {
        /* Dense tensor contiguous data. */
        void *dense;

        /* Data representation of sparse tensor. */
        fang_ten_sparse_coo_t *sparse;
    } data;
} fang_ten_t;

/* Structure to pass dimension data to the Tensor. */
typedef struct fang_ten_dim {
    uint32_t *dims;
    int ndims;
} fang_ten_dim_t;

/* Pass arguments to the operator functions with this structure. */
/* NOTE: Argument structure also can be used to retrieve data from operator
 *   functions. Recommended field for that is `y`.
 */
typedef struct fang_ten_ops_arg {
    /* May used to pass resulting tensor. */
    fang_gen_t dest;

    /* May used to pass input tensors and/or data. */
    fang_gen_t x;
    fang_gen_t y;
    fang_gen_t z;
    fang_gen_t alpha;
    fang_gen_t beta;
} fang_ten_ops_arg_t;

/* Signature of an operator functions. */
typedef int (*fang_ten_operator_fn)(fang_ten_ops_arg_t *restrict arg);

/* Operators supported by Tensor implementation. */
typedef struct fang_ten_ops {
    fang_ten_operator_fn create;
    fang_ten_operator_fn print;
    fang_ten_operator_fn rand;
    fang_ten_operator_fn sum;
    fang_ten_operator_fn diff;
    fang_ten_operator_fn mul;
    fang_ten_operator_fn gemm;
    fang_ten_operator_fn scale;
    fang_ten_operator_fn fill;
    fang_ten_operator_fn release;
} fang_ten_ops_t;

/* Used to indicate whether to transpose a matrix (two trailing dimension within
   tensor) while performing GEMM. */
typedef enum fang_ten_gemm_transp {
    FANG_TEN_GEMM_NO_TRANSPOSE,
    FANG_TEN_GEMM_TRANSPOSE
} fang_ten_gemm_transp_t;

/* ================ DATA STRUCTURES END ================ */


/* ================ DECLARATIONS ================ */

/* Creates a new dense tensor. */
FANG_API FANG_HOT int fang_ten_create(fang_ten_t *ten, int eid,
    fang_ten_dtype_t dtyp, fang_ten_dim_t dim, void *restrict data);

/* Creates a scalar tensor. */
FANG_API FANG_HOT int fang_ten_scalar(fang_ten_t *ten, int eid,
    fang_ten_dtype_t dtyp, fang_gen_t value);

/* Prints a tensor to a file. */
FANG_API FANG_HOT int fang_ten_fprint(fang_ten_t *ten, const char *name,
    int padding, FILE *file);

/* Fill dense tensor with random numbers. */
FANG_API int fang_ten_rand(fang_ten_t *ten, fang_gen_t low, fang_gen_t high,
    uint32_t seed);

/* Scales a tensor. */
FANG_API FANG_HOT int fang_ten_scale(fang_ten_t *ten, fang_gen_t factor);

/* Fills the tensor with given value. */
FANG_API FANG_HOT int fang_ten_fill(fang_ten_t *ten, fang_gen_t value);

/* Adds two tensor. */
FANG_API FANG_HOT int fang_ten_sum(fang_ten_t *dest, fang_ten_t *x,
    fang_ten_t *y);

/* Subtracts two tensor. */
FANG_API FANG_HOT int fang_ten_diff(fang_ten_t *dest, fang_ten_t *x,
    fang_ten_t *y);

/* Element-wise multiplies two tensor. */
FANG_API FANG_HOT int fang_ten_mul(fang_ten_t *dest, fang_ten_t *x,
    fang_ten_t *y);

/* Performs General Matrix-Matrix Multiply (GEMM) operation on two trailing
   dimension. */
/* dest := alpha * xy + beta * dest */
FANG_API FANG_HOT int fang_ten_gemm(fang_ten_gemm_transp_t transp_x,
    fang_ten_gemm_transp_t transp_y, fang_gen_t beta, fang_ten_t * dest,
    fang_gen_t alpha, fang_ten_t *x, fang_ten_t *y);

// TODO: Add fang_ten_fma (Fuse Multiply-Add) using FMA extension

/* Releases a tensor. */
FANG_API FANG_HOT int fang_ten_release(fang_ten_t *ten);

/* ================ DECLARATIONS END ================ */


/* ================ INLINE DEFINITIONS ================ */

/* Performs matrix-multiplication between two tensors. It's just a wrapper
 * around `fang_ten_gemm()`, which is much more genralized. */
FANG_HOT FANG_INLINE static inline int fang_ten_matmul(fang_ten_t *dest,
    fang_ten_t *x, fang_ten_t *y)
{
    return fang_ten_gemm(FANG_TEN_GEMM_NO_TRANSPOSE, FANG_TEN_GEMM_NO_TRANSPOSE,
        FANG_F2G(0.0f), dest, FANG_F2G(1.0f), x, y);
}

/* ================ INLINE DEFINITIONS END ================ */


#endif  // FANG_TENSOR_H
