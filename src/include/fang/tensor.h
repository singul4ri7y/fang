#ifndef FANG_TENSOR_H
#define FANG_TENSOR_H

#include <fang/type.h>

/* ================ DATA STRUCTURES ================ */

/* Data type of each tensor. */
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
    FANG_TEN_DTYPE_FLOAT32,
    FANG_TEN_DTYPE_FLOAT64
} fang_ten_dtype_t;

/* Represents a single tensor. */
typedef struct fang_ten {
    /* Environment ID with which the tensor is created. */
    uint16_t eid;

    /* The number of dimensions. */
    uint16_t ndims;

    /* Type of data tensor is holding. */
    fang_ten_dtype_t dtyp;

    /* The dimension and stride is stored simultaneously, where individual
       dimension can be retrieved through a single division operation. */
    /* Let's call it "stridemension" B). */
    uint32_t *sdims;

    /* The contiguous data. */
    void *data;
} fang_ten_t;

/* Pass arguments to the operator functions with this structure. */
typedef struct fang_ten_ops_arg {
    /* May used to pass resulting tensor. */
    fang_gen dest;

    /* May used to pass input tensors. */
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

#endif  // FANG_TENSOR_H
