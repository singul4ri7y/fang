#include <fang/tensor.h>
#include <fang/platform.h>
#include <fang/status.h>
#include <stdio.h>
#include <string.h>

/* ---------------- DEFINITIONS ---------------- */ 

/* Creates and returns a new tensor. */
int fang_ten_create(fang_ten_t *restrict ten, int pid, fang_ten_dtype_t typ, 
    uint32_t *restrict dims, uint16_t ndims, void *restrict data) 
{
    int res = FANG_GENOK;

    // TODO: Handle cases of scalar tensors (0-dimensional tensors).

    if(ndims == 0) {
        res = -FANG_INVDIM;
        goto out;
    }

    /* We need a valid platform. */
    fang_platform_t *plat;
    if(!FANG_OK(res = _fang_platform_get(&plat, pid))) 
        goto out;

    /* This is a very unusual way to store strides and the 
       dimensions simultaneously. */
    /* This is for reasonable computaton-memory tradeoff. */
    /* Usually the 'sdims' field stores the stride of each dimensions, but
       in this case, the very last slot will hold the very first dimension.
       Rest of the slots will be used for storing the strides. */
    ten->ndims = ndims;
    ten->sdims = FANG_CREATE(plat->realloc, ndims * sizeof(*ten->sdims));

    if(ten->sdims == NULL) {
        res = -FANG_NOMEM;
        goto out;
    }

    /* Initial final slot is 1. It is a pure stride array for now. */ 
    ten->sdims[ndims - 1] = 1;

    for(int i = ndims - 2; i >= 0; i--) 
        ten->sdims[i] = dims[i + 1] * ten->sdims[i + 1];

    /* Hold the first dimension and calculate total size. */
    size_t size = ten->sdims[0];
    ten->sdims[ndims - 1] = dims[0];
    size *= dims[0];

    size_t ndtyp = 0;
    switch(typ) {
        case FANG_TEN_DTYPE_FLOAT64:
        case FANG_TEN_DTYPE_INT64:
        case FANG_TEN_DTYPE_UINT64: 
            ndtyp = 8;
            break;

        case FANG_TEN_DTYPE_FLOAT32: 
        case FANG_TEN_DTYPE_INT32:
        case FANG_TEN_DTYPE_UINT32:
            ndtyp = 4;
            break;

        case FANG_TEN_DTYPE_INT16: 
        case FANG_TEN_DTYPE_UINT16: 
            ndtyp = 2;
            break;

        case FANG_TEN_DTYPE_INT8: 
        case FANG_TEN_DTYPE_UINT8: 
            ndtyp = 1;
            break;

        default: {
            res = -FANG_INVTYP;
            goto out;
        }
    }

    fang_ten_ops_arg_t arg = {
        .typ = typ,
        .size = size,
        .plat = plat,
        /* We will do something different here. Instead of passing our tensor data
           here, we will be passing the initializer data. */
        .data = data
    };

    if(!FANG_OK(res = plat->ops->create(&arg, &ten->data, ndtyp))) 
        goto out;

    /* We have a valid type. */
    ten->dtyp = typ;

    /* Set the platform id. */
    ten->pid = pid;

    /* We successfully created a tensor! */
    plat->ntens++;

out: 
    return res;
}

/* Releases the tensor. */
int fang_ten_release(fang_ten_t *restrict ten) {
    int res = FANG_GENOK;

    /* We need a valid platform. */
    fang_platform_t *plat;
    if(!FANG_OK(res = _fang_platform_get(&plat, ten->pid))) 
        goto out;

    FANG_RELEASE(plat->realloc, ten->sdims);

    /* Now release the data. */
    fang_ten_ops_arg_t arg = { .plat = plat, .data = ten->data };
    plat->ops->release(&arg);

    /* Update the tensor count. */
    plat->ntens--;

out: 
    return res;
}

/* Prints a tensor. */
int fang_ten_print(fang_ten_t *restrict ten, const char *name, int padding) {
    int res = FANG_GENOK;

    fang_platform_t *plat;
    if(!FANG_OK(res = _fang_platform_get(&plat, ten->pid))) 
        goto out;

    printf("%*s[%s] = \n", padding, "", name);

    /* We pass the stridemension of the tensor in the arg structure
       and pass the tensor data separately. */
    fang_ten_ops_arg_t arg = {
        .typ  = ten->dtyp,
        .size = ten->ndims,
        .plat = (void *) plat,
        .data = (void *) ten->sdims
    };
    res = plat->ops->print(&arg, ten->data, padding);

out: 
    return res;
}

/* Randomize the entire tensor. */
int fang_ten_rand(fang_ten_t *restrict ten, void *low, void *high) {
    int res = FANG_GENOK;

    fang_platform_t *plat;
    if(!FANG_OK(res = _fang_platform_get(&plat, ten->pid))) 
        goto out;

    fang_ten_ops_arg_t arg = {
        .typ  = ten->dtyp,
        .size = ten->sdims[0] * ten->sdims[ten->ndims - 1],
        .plat = (void *) plat,
        .data = ten->data
    };
    res = plat->ops->rand(&arg, low, high);

out: 
    return res;
}

#define FANG_TEN_ARITH(type)                                           \
int fang_ten_##type(fang_ten_t *dest, fang_ten_t *a, fang_ten_t *b) {  \
    int res = FANG_GENOK;                                              \
    /* Are the tensors belong to same platform? */                     \
    if(dest->pid != a->pid || a->pid != b->pid) {                      \
        res = -FANG_INVPL;                                             \
        goto out;                                                      \
    }                                                                  \
    /* Are the tensors holding same type? */                           \
    if(dest->dtyp != a->dtyp || a->dtyp != b->dtyp) {                  \
        res = -FANG_INVTYP;                                            \
        goto out;                                                      \
    }                                                                  \
    /* Check whether we have same dimensions across tensors. */        \
    if(dest->ndims != a->ndims || a->ndims != b->ndims) {              \
        res = -FANG_INVDIM;                                            \
        goto out;                                                      \
    }                                                                  \
    if(memcmp(dest->sdims, a->sdims, a->ndims * sizeof(uint32_t))      \
        || memcmp(a->sdims, b->sdims, a->ndims * sizeof(uint32_t)))    \
    {                                                                  \
        res = -FANG_INVDIM;                                            \
        goto out;                                                      \
    }                                                                  \
    fang_platform_t *plat;                                             \
    if(!FANG_OK(res = _fang_platform_get(&plat, a->pid)))              \
        goto out;                                                      \
    fang_ten_ops_arg_t arg = {                                         \
        .typ  = a->dtyp,                                               \
        .size = a->sdims[0] * a->sdims[a->ndims - 1],                  \
        .plat = (void *) plat,                                         \
        .data = dest->data                                             \
    };                                                                 \
    res = plat->ops->type(&arg, a->data, b->data);                     \
out:                                                                   \
    return res;                                                        \
}

/* Performs summation operation between two tensors and stores it in destination
   tensor. */
/* Returns the destination tensor. */
FANG_TEN_ARITH(sum);

/* Performs subtraction operation between two tensors and stores it in destination
   tensor. */
/* Returns the destination tensor. */
FANG_TEN_ARITH(diff);

/* Performs element-wise multiplication (Hadamard product) between two tensors 
   and stores it in destination tensor. */
/* Returns the destination tensor. */
FANG_TEN_ARITH(hadamard);

/* ---------------- DEFINITIONS END ---------------- */ 
