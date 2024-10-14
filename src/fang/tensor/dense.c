#include <fang/util/buffer.h>
#include <fang/tensor.h>
#include <fang/env.h>
#include <fang/status.h>
#include <compiler.h>
#include <string.h>
#include <stdbool.h>

/* ================ HELPER MACROS ================ */

/* Self explanatory. */
#define _FANG_MAX(x, y)         (x > y ? x : y)
#define _FANG_MIN(x, y)         (x < y ? x : y)

/* ================ HELPER MACROS ================ */


/* ================ PRIVATE DEFINITIONS ================ */

/* Calculates strides. */
FANG_HOT FANG_INLINE static inline void
    _fang_ten_calc_strides(uint32_t *strides, uint32_t *dims, int ndims)
{
    strides[ndims - 1] = 1;
    for(int i = ndims - 2; i >= 0; i--)
        strides[i] = dims[i + 1] * strides[i + 1];
}

/* There are 5 broadcast patterns:
 * 1. Boradcast scalar against N-dimensional tensor. Denoted with 1.
 * 2. Broadcast vector against N-dimensional tensor. Denoted with 2.
 * 3. Broadcast a single channel. E.g. while operating over a (2, 3, 5, 3) and
 *    (1, 1, 1, 3) tensors. Works just like adding a vector. Also denoted by 2.
 * 4. Broadcast a matrix against N-dimentional tensor (array of
 *    matrices of same shape). Denoted with 3.
 * 5. Broadcast dimension is unknown. Denoted with 4.
 * 6. No broadcasting :). Denoted with 0.
 */
FANG_HOT FANG_INLINE static inline int
    _fang_broadcast_pattern(fang_ten_t *restrict pwx, fang_ten_t *restrict pwy)
{
    /* Broadcast dimension unknown. */
    int pattern = 4;

    /* Swap large tensor in terms of size. */
    if(FANG_UNLIKELY(pwy->strides[0] * pwy->dims[0] >
        pwx->strides[0] * pwx->dims[0]))
    {
        fang_ten_t temp = *pwy;
        *pwy = *pwx;
        *pwx = temp;
    }

    /* Scalar tensor/1-element tensors against N-dimensional tensor. */
    if(FANG_LIKELY(pwy->strides[0] * pwy->dims[0] == 1))
        pattern = 1;
    /* Vector against N-dimensional tensor. */
    else if(FANG_LIKELY(pwy->ndims == 1 &&
        pwx->dims[pwx->ndims - 1] == pwy->dims[0]))
    {
        pattern = 2;
    }
    /* Broadcast over a channel. Mostly used when adding bias in
       image processing. */
    else if(FANG_LIKELY(pwy->strides[0] * pwy->dims[0] ==
        pwx->dims[pwx->ndims - 1]))
    {
        pattern = 2;
    }
    /* Matrix against N-dimensional matrices. */
    else if(FANG_LIKELY(pwy->ndims == 2 && !memcmp(pwx->dims + (pwx->ndims - 2),
        pwy->dims + (pwy->ndims - 2), 2 * sizeof(*pwx->dims))))
    {
        pattern = 3;
    }

    return pattern;
}

/* ================ PRIVATE DEFINITIONS END ================ */


/* ================ DEFINITIONS ================ */

/* Creates a new dense tensor. */
int fang_ten_create(fang_ten_t *ten, int eid, fang_ten_dtype_t dtyp,
    uint32_t *restrict dims, uint16_t ndims, void *restrict data)
{
    int res = FANG_OK;

    /* Is passed dimension valid? */
    if(FANG_UNLIKELY(ndims == 0 || dims == NULL)) {
        res = -FANG_INVDIM;
        goto out;
    }

    /* Is tensor data type valid? */
    if(FANG_UNLIKELY(dtyp < FANG_TEN_DTYPE_INT8 &&
        dtyp > FANG_TEN_DTYPE_FLOAT64))
    {
        res = -FANG_INVDTYP;
        goto out;
    }

    /* Tensor type and data type. */
    ten->typ  = FANG_TEN_TYPE_DENSE;
    ten->dtyp = dtyp;

    /* Retrieve Environment structure. */
    fang_env_t *env;
    if(FANG_UNLIKELY(!FANG_ISOK(res = _fang_env_retrieve(&env, eid))))
        goto out;

    /* Validate dimensions. */
    for(int i = 0; i < ndims; i++) {
        if(FANG_UNLIKELY(dims[i] == 0)) {
            res = -FANG_INVDIM;
            goto out;
        }
    }
    /* Store dimensions. */
    ten->ndims = ndims;
    ten->dims = FANG_CREATE(env->realloc, *ten->dims, ndims);
    if(ten->dims == NULL) {
        res = -FANG_NOMEM;
        goto out;
    }
    memcpy(ten->dims, dims, ndims * sizeof(*ten->dims));

    /* Calculate and store strides. */
    ten->strides = FANG_CREATE(env->realloc, *ten->strides, ndims);
    if(ten->strides == NULL) {
        res = -FANG_NOMEM;
        goto out;
    }
    _fang_ten_calc_strides(ten->strides, dims, ndims);

    /* Call operator. */
    fang_ten_ops_arg_t arg = {
        .dest = (fang_gen *) ten,

        /* Number of elements. */
        .x = FANG_U2G((fang_uint) ten->strides[0] * dims[0]),
        .y = (fang_gen) data,
        .z = (fang_gen) env
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
int fang_ten_scalar(fang_ten_t *ten, int eid, fang_ten_dtype_t dtyp,
    fang_gen value)
{
    int res = FANG_OK;

    /* Is tensor data type valid? */
    if(FANG_UNLIKELY(dtyp < FANG_TEN_DTYPE_INT8 &&
        dtyp > FANG_TEN_DTYPE_FLOAT64))
    {
        res = -FANG_INVDTYP;
        goto out;
    }

    /* Check whether the Environment is valid. */
    fang_env_t *env;
    if(FANG_UNLIKELY(!FANG_ISOK(res = _fang_env_retrieve(&env, eid))))
        goto out;

    /* Fill in the tensor structure. */
    ten->typ     = FANG_TEN_TYPE_DENSE;
    ten->dtyp    = dtyp;
    ten->dims    = NULL;
    ten->strides = NULL;
    ten->ndims   = 0;

    /* Scalar tensors act like single element 1-dimensional tensor. */
    /* `fang_gen` is bitcasted form data types like `fang_float` or `fang_int`.
       Hence, it can be used as generic representation of all the types. */
    fang_gen data[1] = { value };

    fang_ten_ops_arg_t arg = {
        .dest = (fang_gen) ten,
        .y = (fang_gen) data,
        .z = (fang_gen) env
    };
    if(FANG_UNLIKELY(!FANG_ISOK(res = env->ops->dense->create(&arg))))
        goto out;

    /* Tensor creation successful. */
    ten->eid = eid;
    env->ntens++;

out:
    return res;
}

/* Prints a tensor. */
FANG_API FANG_HOT int fang_ten_fprint(fang_ten_t *ten, const char *name,
    int padding, FILE *file)
{
    int res = FANG_OK;

    if(FANG_UNLIKELY(ten->dtyp < FANG_TEN_DTYPE_INT8 &&
        ten->dtyp > FANG_TEN_DTYPE_FLOAT64))
    {
        res = -FANG_INVDTYP;
        goto out;
    }

    fang_env_t *env;
    if(FANG_UNLIKELY(!FANG_ISOK(res = _fang_env_retrieve(&env, ten->eid))))
        goto out;

    /* Print tensor details. */
    /* Padding is useful when printing nn-like structures. */
    fprintf(file, "%*s[%s] = ", padding, "", name);

    /* Do not go newline if single dimension or scalar tensor is printed. */
    if(FANG_LIKELY(ten->ndims != 1 && ten->dims != NULL))
        fprintf(file, "\n");

    /* Buffer to push strings to. */
    fang_buffer_t buff;
    if(FANG_UNLIKELY(!FANG_ISOK(res =
        FANG_BUFFER_CREATE(&buff, env->realloc, char))))
    {
        goto out;
    }

    fang_ten_ops_arg_t arg = {
        .dest = (fang_gen) ten,
        .x = FANG_I2G(padding),
        .y = (fang_gen) &buff
    };
    if(FANG_LIKELY(ten->typ == FANG_TEN_TYPE_DENSE)) {
        if(FANG_UNLIKELY(!FANG_ISOK(res = env->ops->dense->print(&arg))))
            goto release;
    }
    else if(FANG_LIKELY(ten->typ == FANG_TEN_TYPE_SPARSE)) {
        if(FANG_UNLIKELY(!FANG_ISOK(res = env->ops->sparse->print(&arg))))
            goto release;
    } else {
        res = -FANG_INVTENTYP;
        goto release;
    }

    /* Print contents. */
    fprintf(file, "%s\n", (char *) fang_buffer_retrieve(&buff, NULL));

release:
    /* Release the buffer. */
    fang_buffer_release(&buff);

out:
    return res;
}

/* Fill dense tensor with random numbers. */
int fang_ten_rand(fang_ten_t *ten, fang_gen low, fang_gen high, uint32_t seed) {
    int res = FANG_OK;

    /* Tensor has to be dense tensor. */
    if(FANG_UNLIKELY(ten->typ != FANG_TEN_TYPE_DENSE)) {
        res = -FANG_INVTENTYP;
        goto out;
    }

    /* Handle scalar tensor. */
    fang_ten_t input = *ten;
    input.dims    = input.dims == NULL ? (uint32_t []) { 1 } : input.dims;
    input.strides = input.strides == NULL ? (uint32_t []) { 1 } : input.strides;

    /* Get Environment. */
    fang_env_t *env;
    if(FANG_UNLIKELY(!FANG_ISOK(res = _fang_env_retrieve(&env, ten->eid))))
        goto out;

    fang_ten_ops_arg_t arg = {
        .dest = (fang_gen) &input,
        .x = low,
        .y = high,
        .z = FANG_U2G(seed)
    };
    res = env->ops->dense->rand(&arg);

out:
    return res;
}

/* Adds two tensor. */
FANG_API FANG_HOT int fang_ten_sum(fang_ten_t *dest, fang_ten_t *x,
    fang_ten_t *y)
{
    int res = FANG_OK;

    /* Tensors have to belong to same Environment. */
    if(FANG_UNLIKELY(dest->eid != x->eid || x->eid != y->eid)) {
        res = -FANG_ENVMISMATCH;
        goto out;
    }

    /* Tensors have to have same data type. */
    if(FANG_UNLIKELY(dest->dtyp != x->dtyp || x->dtyp != y->dtyp)) {
        res = -FANG_INVDTYP;
        goto out;
    }
    /* Get Environment. */
    fang_env_t *env;
    if(FANG_UNLIKELY(!FANG_ISOK(res = _fang_env_retrieve(&env, x->eid))))
        goto out;

    /* Modified (maybe) version of the tensors */
    fang_ten_t wx = *x, wy = *y, wd = *dest;

    /* Handle scalar tensors. */
    wx.dims = wx.dims == NULL ? (uint32_t []) { 1 } : wx.dims;
    wy.dims = wy.dims == NULL ? (uint32_t []) { 1 } : wy.dims;
    wx.strides = wx.strides == NULL ? (uint32_t []) { 1 } : wx.strides;
    wy.strides = wy.strides == NULL ? (uint32_t []) { 1 } : wy.strides;

    /* Argument structure. */
    fang_ten_ops_arg_t arg = {
        .dest = (fang_gen) &wd,
        .x = (fang_gen) &wx,
        .y = (fang_gen) &wy
    };

    /* Batch system. */
    if(FANG_LIKELY(x->ndims == y->ndims)) {
        /* Check if tensors are broadcastable, if not batched. */
        if(FANG_LIKELY(!memcmp(x->dims, y->dims, x->ndims *
            sizeof(uint32_t))))
        {
            /* Check if destination tensor is valid to store result. */
            if(FANG_UNLIKELY(dest->ndims != x->ndims ||
                memcmp(dest->dims, x->dims, x->ndims * sizeof(uint32_t))))
            {
                res = -FANG_DESTINVDIM;
                goto out;
            }

            /* No need for broadcasting. */
            arg.z = FANG_I2G(0);
            res = env->ops->dense->sum(&arg);
        } else {
            for(int i = 0; i < x->ndims; i++) {
                uint32_t xdim = wx.dims[i];
                uint32_t ydim = wy.dims[i];

                /* Not broadcastable. */
                if(FANG_UNLIKELY(xdim != ydim && xdim != 1 && ydim != 1)) {
                    res = -FANG_NOBROAD;
                    goto out;
                }
            }

            /* Check if destination tensor is valid to store result. */
            if(FANG_UNLIKELY(dest->ndims != x->ndims)) {
                res = -FANG_DESTINVDIM;
                goto out;
            }
            for(int i = 0; i < x->ndims; i++) {
                if(FANG_UNLIKELY(dest->dims[i] !=
                    _FANG_MAX(x->dims[i], y->dims[i])))
                {
                    res = -FANG_DESTINVDIM;
                    goto out;
                }
            }

            /* Calculate the broadcasted strides. */
            uint32_t x_broadcasted_strides[x->ndims];
            uint32_t y_broadcasted_strides[y->ndims];
            int pattern = _fang_broadcast_pattern(&wx, &wy);

            if(FANG_UNLIKELY(pattern == 4)) {
                /* Pre-compute broadcasted strides. */
                for(int i = 0; i < wx.ndims; i++) {
                    uint32_t xdim = wx.dims[i], ydim = wy.dims[i];

                    x_broadcasted_strides[i] = xdim != 1 ? wx.strides[i] : 0;
                    y_broadcasted_strides[i] = ydim != 1 ? wy.strides[i] : 0;
                }

                /* Change strides to broadcasted strides. */
                wx.strides = x_broadcasted_strides;
                wy.strides = y_broadcasted_strides;
            }

            arg.z = FANG_I2G(pattern);
            res = env->ops->dense->sum(&arg);
        }
    } else {
        /* Classify tensors based on max/min dimension count. */
        int dmax = _FANG_MAX(x->ndims, y->ndims),
            dmin = _FANG_MIN(x->ndims, y->ndims);
        int diff = dmax - dmin;

        /* Ensure `wx` always holds tensor with maximum dimension count. */
        if(FANG_UNLIKELY(wx.ndims < wy.ndims)) {
            fang_ten_t temp = wy;
            wy = wx;
            wx = temp;
        }

        /* Check if tensors are broadcastable. */
        for(int i = 0; i < dmin; i++) {
            uint32_t wxdim = wx.dims[i + diff];
            uint32_t wydim = wy.dims[i];

            /* Not broadcastable. */
            if(FANG_UNLIKELY(wxdim != wydim && wxdim != 1 && wydim != 1)) {
                res = -FANG_NOBROAD;
                goto out;
            }
        }

        int pattern = _fang_broadcast_pattern(&wx, &wy);
        uint32_t x_broadcasted_strides[dmax];
        uint32_t y_broadcasted_strides[dmax];

        if(FANG_UNLIKELY(pattern == 4)) {
            /* Calculate broadcasted strides. */
            for(int i = 0; i < dmax; i++) {
                x_broadcasted_strides[i] = wx.dims[i] != 1 ? wx.strides[i] : 0;
                y_broadcasted_strides[i] = i - diff < 0 ? 0 :
                    (wy.dims[i - diff] != 1 ? wy.strides[i - diff] : 0);
            }

            wx.strides = x_broadcasted_strides;
            wy.strides = y_broadcasted_strides;
        }

        arg.z = FANG_I2G(pattern);
        res = env->ops->dense->sum(&arg);
    }

out:
    return res;
}

/* Releases a tensor. */
int fang_ten_release(fang_ten_t *ten) {
    int res = FANG_OK;

    /* Valid Environment expected. */
    fang_env_t *env;
    if(FANG_UNLIKELY(!FANG_ISOK(res = _fang_env_retrieve(&env, ten->eid))))
        goto out;

    /* Release dimension and stride array. */
    FANG_RELEASE(env->realloc, ten->dims);
    FANG_RELEASE(env->realloc, ten->strides);

    /* Now release the data. */
    fang_ten_ops_arg_t arg = {
        .dest = (fang_gen) ten,
        .z = (fang_gen) env
    };
    if(FANG_LIKELY(ten->typ == FANG_TEN_TYPE_DENSE)) {
        if(FANG_UNLIKELY(!FANG_ISOK(res =
            env->ops->dense->release(&arg))))
        {
            goto out;
        }
    }
    else if(FANG_LIKELY(ten->typ == FANG_TEN_TYPE_SPARSE)) {
        if(FANG_UNLIKELY(!FANG_ISOK(res =
            env->ops->sparse->release(&arg))))
        {
            goto out;
        }
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
