#include <fang/tensor.h>
#include <fang/platform.h>
#include <fang/status.h>

int fang_ten_create(fang_ten_t *restrict ten, int pid, uint32_t *restrict dims, 
    uint16_t ndims, void *restrict data) 
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
    ten -> ndims = ndims;
    ten -> sdims = FANG_CREATE(plat -> realloc, ndims * sizeof(*ten -> sdims));

    if(ten -> sdims == NULL) {
        res = -FANG_NOMEM;
        goto out;
    }

    /* Initial final slot is 1. It is a pure stride array for now. */ 
    ten -> sdims[ndims - 1] = 1;

    for(int i = ndims - 2; i >= 0; i--) 
        ten -> sdims[i] = dims[i] * ten -> sdims[i + 1];

    /* Hold the first dimension and calculate total size. */
    size_t size = ten -> sdims[0];
    ten -> sdims[ndims - 1] = dims[0];
    size *= dims[0];
    
    if(!FANG_OK(res = plat -> ops -> create(plat, &ten -> data, size, data))) 
        goto out;

    /* Set the platform id. */
    ten -> pid = pid;

    /* We successfully created a tensor! */
    plat -> ntens++;

out: 
    return res;
}

int fang_ten_release(fang_ten_t *ten) {
    int res = FANG_GENOK;

    /* We need a valid platform. */
    fang_platform_t *plat;
    if(!FANG_OK(res = _fang_platform_get(&plat, ten -> pid))) 
        goto out;

    FANG_RELEASE(plat -> realloc, ten -> sdims);

    /* Now release the data. */
    plat -> ops -> release(plat, ten -> data);

    /* Update the tensor count. */
    plat -> ntens--;

out: 
    return res;
}
