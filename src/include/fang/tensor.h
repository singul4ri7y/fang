#ifndef FANG_TENSOR_H
#define FANG_TENSOR_H

#include <stdint.h>
#include <stddef.h>

/* Tensor print helper macro. */
#define FANG_TEN_PRINT(ten)    fang_ten_print(ten, #ten, 0)

/* ---------------- DATA STRUCTURES ---------------- */

/*
 * Note: If tensor type is any of the floating type, 
 * any input data will be considered of type 'fang_float'.
 *
 * If tensor type is any of the signed/unsigned integer
 * type, the input data will be considered 'fang_(u)int'
 * depending on the signedness.
 *
 * This is applicable to all the external interaction with 
 * tensors, e.g. creating an input tensor with external data.
 */

/* Most precised floating type Fang supports. */
typedef double fang_float;

/* Largest integer type Fang supports. */
typedef int64_t fang_int;
typedef uint64_t fang_uint;

/* Datatype of each tensor. */
typedef enum fang_ten_dtype {
    FANG_TEN_DTYPE_FLOAT64,
    FANG_TEN_DTYPE_FLOAT32,
    FANG_TEN_DTYPE_INT64,
    FANG_TEN_DTYPE_UINT64,
    FANG_TEN_DTYPE_INT32,
    FANG_TEN_DTYPE_UINT32,
    FANG_TEN_DTYPE_INT16,
    FANG_TEN_DTYPE_UINT16,
    FANG_TEN_DTYPE_INT8,
    FANG_TEN_DTYPE_UINT8,
    FANG_TEN_DTYPE_INVALID
} fang_ten_dtype_t;

/* Representation of a single Tensor. */
typedef struct fang_ten {
    /* The Platform ID with which the tensor is created. */
    uint16_t pid;

    /* The number of dimensions. */
    uint16_t ndims;

    /* What type of data we are dealing with in this tensor? */
    fang_ten_dtype_t dtyp;

    /* Here we store both dimension and stride at once. */
    /* Let's call it "stridemension" B). */
    uint32_t *sdims;

    /* The contiguous data. */
    void *data;
} fang_ten_t;

/* Structure to pass as an argument to operational functions. */ 
typedef struct fang_ten_ops_arg {
    /* Type of the tensor in hand. */
    fang_ten_dtype_t typ;

    /* Size of the elements (not in bytes). */
    size_t size;

    /* The platform structure. */
    void *plat;

    /* The contiguous data. */
    void *data;
} fang_ten_ops_arg_t;

/* Tensor operations specific to each platforms. */
typedef struct fang_ten_ops {
    /* Creates a tensor on a specific platform. */
    int  (*create)(fang_ten_ops_arg_t *restrict arg, void **restrict dest, 
            size_t ndtyp);

    /* Releases a tensor on a specific platform. */
    void (*release)(fang_ten_ops_arg_t *restrict arg);
} fang_ten_ops_t;

/* ---------------- DATA STRUCTURES END ---------------- */

/* ---------------- DECLARATIONS ---------------- */

/* Creates and returns a new tensor. */
int fang_ten_create(fang_ten_t *restrict ten, int pid, fang_ten_dtype_t typ,
        uint32_t *restrict dims, uint16_t ndims, void *restrict data);

/* Releases the tensor. */
int fang_ten_release(fang_ten_t *ten);

/* Prints a tensor. */
int fang_ten_print(fang_ten_t *ten, const char *name, int padding);

/* ---------------- DECLARATIONS END ---------------- */

#endif    // FANG_TENSOR_H
