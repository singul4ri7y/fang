#ifndef FANG_TENSOR_H
#define FANG_TENSOR_H

#include <fang/type.h>
#include <stddef.h>

/* ---------------- DATA STRUCTURES ---------------- */

/* Datatype of each tensor. */
typedef enum fang_ten_dtype {
    FANG_TEN_DTYPE_FLOAT64,
    FANG_TEN_DTYPE_FLOAT32,
    FANG_TEN_DTYPE_UINT64,
    FANG_TEN_DTYPE_UINT32,
    FANG_TEN_DTYPE_UINT16,
    FANG_TEN_DTYPE_UINT8,
    FANG_TEN_DTYPE_INT64,
    FANG_TEN_DTYPE_INT32,
    FANG_TEN_DTYPE_INT16,
    FANG_TEN_DTYPE_INT8,
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

    /* Prints a tensor. */
    int  (*print)(fang_ten_ops_arg_t *restrict arg, void *restrict data, 
            int padding);

    /* Randomizes the tensor based on datatype. */
    int  (*rand)(fang_ten_ops_arg_t *restrict arg, fang_gen low, 
            fang_gen high);

    /* Performs summation of two tensor data. */
    int  (*sum)(fang_ten_ops_arg_t *restrict arg, void *restrict a,
            void *restrict b);

    /* Finds difference between two tensor data. */
    int  (*diff)(fang_ten_ops_arg_t *restrict arg, void *restrict a,
            void *restrict b);
    
    /* Performs element wise multiplication (Hadamard product) on two
       tensor data. */
    int  (*hadamard)(fang_ten_ops_arg_t *restrict arg, void *restrict a,
            void *restrict b);
} fang_ten_ops_t;

/* ---------------- DATA STRUCTURES END ---------------- */

/* ---------------- DECLARATIONS ---------------- */

/* Creates and returns a new tensor. */
int fang_ten_create(fang_ten_t *restrict ten, int pid, fang_ten_dtype_t typ,
        uint32_t *restrict dims, uint16_t ndims, void *restrict data);

/* Releases the tensor. */
int fang_ten_release(fang_ten_t *restrict ten);

/* Prints a tensor. */
int fang_ten_print(fang_ten_t *restrict ten, const char *name, int padding);

/* Randomize the entire tensor. */
int fang_ten_rand(fang_ten_t *restrict ten, void *low, void *high);

/* Performs summation operation between two tensors and stores it in destination
   tensor. */
/* Returns the destination tensor. */
int fang_ten_sum(fang_ten_t *dest, fang_ten_t *a, fang_ten_t *b);

/* Performs subtraction operation between two tensors and stores it in destination
   tensor. */
/* Returns the destination tensor. */
int fang_ten_diff(fang_ten_t *dest, fang_ten_t *a, fang_ten_t *b);

/* Performs element-wise multiplication (Hadamard product) between two tensors 
   and stores it in destination tensor. */
/* Returns the destination tensor. */
int fang_ten_hadamard(fang_ten_t *dest, fang_ten_t *a, fang_ten_t *b);

/* ---------------- DECLARATIONS END ---------------- */

#endif    // FANG_TENSOR_H
