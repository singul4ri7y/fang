#ifndef FANG_PLATFORM_H
#define FANG_PLATFORM_H

#include <fang/tensor.h>
#include <fang/plat/cpu/cpu.h>

#define FANG_MAX_PLATFORMS    1024

/* What type of platform we are on? */
typedef enum fang_platform_type {
    FANG_PLATFORM_TYPE_CPU,
    FANG_PLATFORM_TYPE_INVALID
} fang_platform_type_t;

/* Structure for a single platform. */
typedef struct fang_platform {
    /* Memory manager for CPU specific (de)allocations. */
    fang_reallocator_t realloc;

    /* Number of tensors created using this platform. */
    _Atomic int ntens;

    /* What type of platform it is? */
    fang_platform_type_t type;

    /* Platform specific data. */
    void *private;

    /* Tensor operations geared toward that platform. */
    fang_ten_ops_t *ops;

    /* To release platform specific resources. In other words, release 
       a platform. */
    void (*release)(void *restrict private, fang_reallocator_t alloc);
} fang_platform_t;

/* ---------------- DECLARATIONS ---------------- */

/* Creates a platform and returns the ID. */
int fang_platform_create(fang_platform_type_t type, 
        fang_reallocator_t realloc);

/* Releases a platform if it's releasable. */
int fang_platform_release(uint16_t pid);

/* ---------------- DECLARATIONS END ---------------- */

/* ---------------- PRIVATE ---------------- */
/* THESE DECLARATIONS ARE PRIVATE AND SHOULD NOT BE USED UNLESS YOU KNOW WHAT
   YOU ARE DOING. */

int _fang_platform_get(fang_platform_t **restrict plat, uint16_t pid);

/* ---------------- PRIVATE END ---------------- */

#endif // FANG_PLATFORM_H
