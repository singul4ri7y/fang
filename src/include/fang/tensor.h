#ifndef FANG_TENSOR_H
#define FANG_TENSOR_H

#include <stdint.h>
#include <stddef.h>

/* ---------------- DATA STRUCTURES ---------------- */

/* Representation of a single Tensor. */
typedef struct fang_ten {
    /* The Platform ID with which the tensor is created. */
    uint16_t pid;

    /* The number of dimensions. */
    uint16_t ndims;

    /* Here we store both dimension and stride at once. */
    /* Let's call it stridemension B). */
    uint32_t *sdims;

    /* The contiguous data. */
    void *data;
} fang_ten_t;

/* Tensor operations specific to each platforms. */
typedef struct fang_ten_ops {
    /* Creates a tensor on a specific platform. */
    int  (*create)(void *restrict platform, void **restrict dest, 
            size_t size, void *restrict data);

    /* Releases a tensor on a specific platform. */
    void (*release)(void *restrict platform, void *restrict data);
} fang_ten_ops_t;

/* ---------------- DATA STRUCTURES END ---------------- */

/* ---------------- DECLARATIONS ---------------- */

/* Creates and returns a new tensor. */
int fang_ten_create(fang_ten_t *restrict ten, int pid, 
        uint32_t *restrict dims, uint16_t ndims, void *restrict data);

/* Releases the tensor. */
int fang_ten_release(fang_ten_t *ten);

/* ---------------- DECLARATIONS END ---------------- */

#endif    // FANG_TENSOR_H
