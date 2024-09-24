#include <fang/tensor.h>
#include <fang/env.h>
#include <fang/status.h>
#include <compiler.h>

/* ================ DEFINITIONS ================ */

/* Creates a new dense tensor. */
int fang_ten_create(fang_ten_t *restrict ten, int eid, fang_ten_dtype_t dtyp,
    uint32_t *restrict dims, uint16_t ndims, void *restrict data)
{
    int res = FANG_OK;

    /* Is passed dimension valid? */
    if(FANG_UNLIKELY(ndims == 0 || dims == NULL)) {
        res = -FANG_INVDIM;
        goto out;
    }

    /* Is tensor data type valid? */
    if(FANG_UNLIKELY(dtyp < FANG_TEN_DTYPE_INT8 && dtyp > FANG_TEN_DTYPE_FLOAT64)) {
        res = -FANG_INVDTYP;
        goto out;
    }

    /* Tensor type and data type. */
    ten->typ  = FANG_TEN_TYPE_DENSE;
    ten->dtyp = dtyp;

    /* Retrieve Environment structure. */
    fang_env_t *env;
    if(!FANG_ISOK(res = _fang_env_retrieve(&env, eid)))
        goto out;

    /* This is very unusual way to store strides and the
       dimensions simultaneously. */
    /* This is for reasonable computation-memory tradeoff. */
    /* Usually the 'sdims' field stores the stride of each dimensions, but
       in this case, the very last slot will hold the very first dimension.
       Rest of the slots will be used for storing the strides. */
    ten->ndims = ndims;
    ten->sdims = FANG_CREATE(env->realloc, *ten->sdims, ndims);
    if(ten->sdims == NULL) {
        res = -FANG_NOMEM;
        goto out;
    }

    /* Initial final slot is 1. It is a pure stride array for now. */
    ten->sdims[ndims - 1] = 1;

    for(int i = ndims - 2; i >= 0; i--)
        ten->sdims[i] = dims[i + 1] * ten->sdims[i + 1];

    /* Hold the first dimension and calculate total size. */
    ten->sdims[ndims - 1] = dims[0];

    /* Call operator. */
    fang_ten_ops_arg_t arg = {
        .dest = (fang_gen *) ten,

        /* Number of elements. */
        .x = FANG_U2G((fang_uint) ten->sdims[0] * dims[0]),
        .y = (fang_gen) data
    };
    if(FANG_UNLIKELY(!FANG_ISOK(res = env->ops->dense->create(&arg))))
        goto out;

    /* Tensor creation successful. */
    ten->eid = eid;
    env->ntens++;

out:
    return res;
}

/* Creates a scalar tensor. */
int fang_ten_scalar(fang_ten_t *restrict ten, int eid, fang_ten_dtype_t dtyp,
    fang_gen value)
{
    int res = FANG_OK;

    /* Is tensor data type valid? */
    if(FANG_UNLIKELY(dtyp < FANG_TEN_DTYPE_INT8 && dtyp > FANG_TEN_DTYPE_FLOAT64)) {
        res = -FANG_INVDTYP;
        goto out;
    }

    /* Check whether the Environment is valid. */
    fang_env_t *env;
    if(FANG_UNLIKELY(res = _fang_env_retrieve(&env, eid)))
        goto out;

    /* Fill in the tensor structure. */
    ten->eid   = eid;
    ten->typ   = FANG_TEN_TYPE_DENSE;
    ten->dtyp  = dtyp;
    ten->sdims = NULL;
    ten->ndims = 0;

    fang_ten_ops_arg_t arg = { .dest = (fang_gen) ten, .y = value };
    if(FANG_UNLIKELY(!FANG_ISOK(res = env->ops->dense->create(&arg))))
        goto out;

out:
    return res;
}

/* Releases a tensor. */
int fang_ten_release(fang_ten_t *restrict ten) {
    int res = FANG_OK;

    /* Valid Environment expected. */
    fang_env_t *env;
    if(FANG_UNLIKELY(!FANG_ISOK(res = _fang_env_retrieve(&env, ten->eid))))
        goto out;

    /* Release stridemension. */
    FANG_RELEASE(env->realloc, ten->sdims);

    /* Now release the data. */
    fang_ten_ops_arg_t arg = { .dest = (fang_gen) ten };
    if(FANG_LIKELY(ten->typ == FANG_TEN_TYPE_DENSE)) {
        if(FANG_UNLIKELY(res = env->ops->dense->release(&arg)))
            goto out;
    }
    else if(FANG_LIKELY(ten->typ == FANG_TEN_TYPE_SPARSE)) {
        if(FANG_UNLIKELY(res = env->ops->sparse->release(&arg)))
            goto out;
    } else {
        res = -FANG_INVTENTYP;
        goto out;
    }

    /* Update tensor count. */
    env->ntens--;

out:
    return res;
}

/* ================ DEFINITIONS ================ */
