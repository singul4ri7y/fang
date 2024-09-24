#ifndef FANG_TENSOR_H
#define FANG_TENSOR_H

#include <fang/config.h>
#include <fang/type.h>
#include <compiler.h>

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

    /* Contiguous non-zero elements. */
    void *data;
} fang_ten_sparse_coo_t;

/* Data type of each tensor. */
/* NOTE: DO NOT MESS AROUND WITH ORDER. */
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
    FANG_TEN_DTYPE_FLOAT64
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

    /* The dimension and stride is stored simultaneously, where individual
       dimension can be retrieved through a single division operation. */
    /* Let's call it "stridemension" B). */
    uint32_t *sdims;

    /* Tensor data. */
    union {
        /* Dense tensor contiguous data. */
        /* Dense tensor will be used to represent scalar tensors as well. */
        fang_gen dense;

        /* Data representation of sparse tensor. */
        fang_ten_sparse_coo_t *sparse;
    } data;
} fang_ten_t;

/* Pass arguments to the operator functions with this structure. */
typedef struct fang_ten_ops_arg {
    /* May used to pass resulting tensor. */
    fang_gen dest;

    /* May used to pass input tensors or data. */
    fang_gen x;
    fang_gen y;
} fang_ten_ops_arg_t;

/* Signature of an operator functions. */
typedef int (*fang_ten_operator_fn)(fang_ten_ops_arg_t *restrict arg);

typedef struct fang_ten_ops {
    fang_ten_operator_fn create;
    fang_ten_operator_fn release;
} fang_ten_ops_t;

/* ================ DATA STRUCTURES END ================ */


/* ================ DECLARATIONS ================ */

/* Creates a new dense tensor. */
FANG_API FANG_HOT int fang_ten_create(fang_ten_t *restrict ten, int eid,
    fang_ten_dtype_t dtyp, uint32_t *restrict dims, uint16_t ndims,
    void *restrict data);

/* Creates a scalar tensor. */
FANG_API FANG_HOT int fang_ten_scalar(fang_ten_t *restrict ten, int eid,
    fang_ten_dtype_t dtyp, fang_gen value);

/* Releases a tensor. */
FANG_API FANG_HOT int fang_ten_release(fang_ten_t *restrict ten);

/* ================ DECLARATIONS END ================ */

#endif  // FANG_TENSOR_H
