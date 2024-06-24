#include <fang/tensor.h>
#include <fang/platform.h>
#include <fang/status.h>
#include <stdio.h>
#include <stdbool.h>

/* ---------------- PRIVATE ---------------- */

static void _fang_ten_print_rec(fang_ten_dtype_t typ, void *restrict data, 
    uint16_t level, uint32_t *restrict sdims, uint16_t ndims, 
    uint32_t *restrict indicies, int padding, bool end) 
{
    if(level == ndims) {
        uint32_t stride = 0;
        for(uint16_t i = 0; i < level; i++) 
            stride += indicies[i] * (i + 1 < level ? sdims[i] : 1);

        putchar(' ');
        switch(typ) {
            case FANG_TEN_DTYPE_FLOAT64: 
                printf("%10.4lf", ((double *) data)[stride]);
                break;

            case FANG_TEN_DTYPE_FLOAT32: 
                printf("%8.3f", ((float *) data)[stride]);
                break;

            case FANG_TEN_DTYPE_INT64: 
                printf("%10ld", ((int64_t *) data)[stride]);
                break;

            case FANG_TEN_DTYPE_UINT64: 
                printf("%10lu", ((uint64_t *) data)[stride]);
                break;

            case FANG_TEN_DTYPE_INT32: 
                printf("%10d", ((int32_t *) data)[stride]);
                break;

            case FANG_TEN_DTYPE_UINT32: 
                printf("%10u", ((uint32_t *) data)[stride]);
                break;

            case FANG_TEN_DTYPE_INT16: 
                printf("%5hd", ((int16_t *) data)[stride]);
                break;

            case FANG_TEN_DTYPE_UINT16: 
                printf("%5hu", ((uint16_t *) data)[stride]);
                break;

            case FANG_TEN_DTYPE_INT8: 
                printf("%3hhd", ((int8_t *) data)[stride]);
                break;

            case FANG_TEN_DTYPE_UINT8: 
                printf("%3hhu", ((uint8_t *) data)[stride]);
                break;

            default: break;
        }

        return;
    }

    uint32_t dim = sdims[level];

    /* In stridemension we store the first dimension at the last
       index. */
    if(level == 0) 
        dim = sdims[ndims - 1];
    else if(level + 1 == ndims) 
        dim = sdims[level - 1];
    else dim = (sdims[level - 1] / sdims[level]);

    if(level == 0) 
        printf("%*s", padding, "");
    else if(level > 0 && indicies[level - 1] != 0) 
        printf("%*s", level + padding, "");

    printf("[");
    /* We are at the final dimension. */
    if(level + 1 == ndims) putchar(' ');

    for(indicies[level] = 0; indicies[level] < dim; indicies[level]++) {
        _fang_ten_print_rec(typ, data, level + 1, sdims, ndims, 
            indicies, padding, (indicies[level] + 1 == dim));
    }

    if(level + 1 == ndims) putchar(' ');
    printf("]");

    /* Are we at the end of previous dimension? */
    if(!end) {
        int count = ndims - level;
        while(count--)
            printf("\n");
    }
}

/* ---------------- PRIVATE END ---------------- */

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
int fang_ten_release(fang_ten_t *ten) {
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
int fang_ten_print(fang_ten_t *ten, const char *name, int padding) {
    int res = FANG_GENOK;

    fang_platform_t *plat;
    if(!FANG_OK(res = _fang_platform_get(&plat, ten->pid))) 
        goto out;

    /* Allocate the indicies. */
    uint32_t *indicies = FANG_CREATE(plat->realloc, ten->ndims * sizeof(uint16_t));

    if(indicies == NULL) {
        res = -FANG_NOMEM;
        goto out;
    }

    printf("%*s[%s] = \n", padding, "", name);

    _fang_ten_print_rec(ten->dtyp, ten->data, 0, ten->sdims, ten->ndims, 
        indicies, padding, false);

out: 
    return res;
}

/* ---------------- DEFINITIONS END ---------------- */ 
