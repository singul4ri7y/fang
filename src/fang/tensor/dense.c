#include <fang/util/buffer.h>
#include <fang/tensor.h>
#include <fang/env.h>
#include <fang/status.h>
#include <compiler.h>
#include <string.h>
#include <stdbool.h>

/* ================ HELPER MACROS ================ */

/* Private to this translation. */
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

/* Returns whether specific broadcasting type being used (fast-route). Check out
 * `tensor.h` for more info. */
FANG_HOT FANG_INLINE static inline int _fang_ten_get_broadcast_pattern(
    fang_ten_t *restrict pwx, fang_ten_t *restrict pwy, int *swapped)
{
    /* Broadcast dimension unknown. */
    int pattern = FANG_BCAST_UNKNOWN;

    /* Size of the broadcastee tensor. */
    uint32_t pwy_siz = pwy->strides[0] * pwy->dims[0];

    /* Swap large tensor in terms of size. Larger tensor at pwx, smaller tensor
     * at pwy. This is done to make sure only the broadcastee tensor stays to
     * the right-hand-side. */
    {
        uint32_t pwx_siz = pwx->strides[0] * pwx->dims[0];
        if(FANG_UNLIKELY(pwy_siz > pwx_siz))
        {
            fang_ten_t temp = *pwy;
            *pwy = *pwx;
            *pwx = temp;

            pwy_siz = pwx_siz;

            /* Tensor got swapped. This may be relevant for subtraction. */
            *swapped ^= 1;
        };
    }

    /* Scalar tensor/1-element tensor against N-dimensional tensor. */
    if(FANG_LIKELY(pwy_siz == 1))
        pattern = FANG_BCAST_SCALAR;
    /* Matrix against N-dimensional matrices. */
    /* `pwy` is a perfect matrix if first stride is equal to matrix
       size or last dimension. */
    /* Multiplying `pwy->dims[0]` ensures leading dimensions are filled with
       ones */
    else if(FANG_LIKELY(pwy->ndims >= 2 &&
        pwx->dims[pwx->ndims - 1] == pwy->dims[pwy->ndims - 1] &&
        pwx->dims[pwx->ndims - 2] == pwy->dims[pwy->ndims - 2] &&
       (pwy->strides[0] == pwy->dims[pwy->ndims - 1] ||
        pwy->strides[0] == pwy->dims[pwy->ndims - 1] *
        pwy->dims[pwy->ndims - 2] * pwy->dims[0])))
    {
        pattern = FANG_BCAST_MATRIX;
    }
    /* Row-major vector against N-dimensional tensor. */
    /* `pwy` is a perfect row vector if first stride is equal to last dimension
       or 1. */
    else if(FANG_LIKELY(
        pwx->dims[pwx->ndims - 1] == pwy->dims[pwy->ndims - 1] &&
       (pwy->strides[0] == 1 || pwy->strides[0] == pwy->dims[pwy->ndims - 1] *
        pwy->dims[0])))
    {
        pattern = FANG_BCAST_ROWVEC;
    }
    /* Col-major vector against N-dimensional tensor. */
    /* `pwy` is a perfect col vector if last dimension is 1 and first stride is
       equal to second last dimension or 1. */
    else if(FANG_LIKELY(pwy->ndims >= 2 && pwy->dims[pwy->ndims - 1] == 1 &&
        pwx->dims[pwx->ndims - 2] == pwy->dims[pwy->ndims - 2] &&
       (pwy->strides[0] == 1 || pwy->strides[0] == pwy->dims[pwy->ndims - 2] *
        pwy->dims[0])))
    {
        pattern = FANG_BCAST_COLVEC;
    }

    return pattern;
}

/* CLARIFICATION: Broadcasting for `fang_ten_gemm()` is done by
 *   considering the operand matrix as a single element. Hence, the operand
 *   matrix is not considered in dimension related computation. */
/* Hence, when it is said, say, "Scalar tensor GEMM against N-dimensional
 * tensor", that would mean a single matrix is being broadcast througout
 * numerous amount of flattened matrices (tensors can be thought as series
 * of flattened matrices) and a single matrix can be thought of as a single
 * scalar element if each operand matrix is considered a single element. */

/* Returns whether specific broadcasting fast-route being used for
 * `fang_ten_gemm()`. Check out `tensor.h` for more info. */
FANG_HOT FANG_INLINE static inline int _fang_ten_gemm_get_broadcast_pattern(
    fang_ten_t *restrict pwx, fang_ten_t *restrict pwy)
{
    /* Broadcast dimension unknown. */
    int pattern = FANG_BCAST_UNKNOWN;

    /* Size of the broadcastee tensor and it's corresponding matrix. */
    uint32_t pwy_matsiz = pwy->dims[pwy->ndims - 1] * pwy->dims[pwy->ndims - 2];
    uint32_t pwy_first_stride = pwy->strides[0] / pwy_matsiz;
    if(pwy_first_stride == 0)  // Dealing with a single operand matrix
        pwy_first_stride = 1;

    uint32_t pwy_siz = pwy_first_stride * pwy->dims[0];

    /* Swap large tensor in terms of size exluding operand matrix. Larger tensor
     * at `pwx`, smaller tensor at `pwy`. This is done to make sure only the
     * broadcastee tensor stays to the right-hand-side. */
    {
        uint32_t pwx_matsiz = pwx->dims[pwx->ndims - 1] *
            pwx->dims[pwx->ndims - 2];
        uint32_t pwx_siz = (pwx->strides[0] * pwx->dims[0]) / pwx_matsiz;

        if(FANG_UNLIKELY(pwy_siz > pwx_siz))
        {
            fang_ten_t temp = *pwy;
            *pwy = *pwx;
            *pwx = temp;

            pwy_siz = pwx_siz;
        };
    }

    /* NOTE: Here, consider a matrix as an unit element. Check for fast-route
     *   keeping that in mind (inherently ignoring the last two dimensions).
     */
    /* Example: Say, two tensors are being GEMMed. One is (3, 5, 3, 4, 6) and
     * another one is (5, 1, 4, 6). During broadcasting, operand matrices are
     * considered units and discarded. So, during broadcasting the tensor
     * becomes (3, 5, 3) and (5, 1) and fast-route is searched. */

    /* Scalar tensor/1-element tensor against N-dimensional tensor. */
    if(FANG_LIKELY(pwy_siz == 1))
        pattern = FANG_BCAST_SCALAR;
    /* Matrix against N-dimensional matrices. */
    /* `pwy` is a perfect broadcastable matrix if first stride is equal to
       broadcastable matrix size of `pwx` or last dimension (excluding the
       actual matrix). */
    /* Multiplying `pwy->dims[0]` ensures leading dimensions are filled with
       one */
    else if(FANG_LIKELY(pwy->ndims >= 4 &&
        pwx->dims[pwx->ndims - 3] == pwy->dims[pwy->ndims - 3] &&
        pwx->dims[pwx->ndims - 4] == pwy->dims[pwy->ndims - 4] &&
       (pwy_first_stride == pwy->dims[pwy->ndims - 3] ||
        pwy_first_stride == pwy->dims[pwy->ndims - 3] *
        pwy->dims[pwy->ndims - 4] * pwy->dims[0])))
    {
        pattern = FANG_BCAST_MATRIX;
    }
    /* Row-major vector against N-dimensional tensor. */
    /* `pwy` is a perfect row vector if first stride is equal to last dimension
       or 1 (considering matrix as an unit element). */
    else if(FANG_LIKELY(
        pwx->dims[pwx->ndims - 3] == pwy->dims[pwy->ndims - 3] &&
       (pwy_first_stride == 1 || pwy_first_stride ==
        pwy->dims[pwy->ndims - 3] * pwy->dims[0])))
    {
        pattern = FANG_BCAST_ROWVEC;
    }
    /* Col-major vector against N-dimensional tensor. */
    /* `pwy` is a perfect col vector if last dimension is 1 and first stride is
       equal to second last dimension or 1 (excluding the matrix). */
    else if(FANG_LIKELY(pwy->ndims >= 4 && pwy->dims[pwy->ndims - 3] == 1 &&
        pwx->dims[pwx->ndims - 4] == pwy->dims[pwy->ndims - 4] &&
       (pwy_first_stride == 1 || pwy_first_stride ==
        pwy->dims[pwy->ndims - 4] * pwy->dims[0])))
    {
        pattern = FANG_BCAST_COLVEC;
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
    memmove(ten->dims, dims, ndims * sizeof(*ten->dims));

    /* Calculate and store strides. */
    ten->strides = FANG_CREATE(env->realloc, *ten->strides, ndims);
    if(ten->strides == NULL) {
        res = -FANG_NOMEM;
        goto out;
    }
    _fang_ten_calc_strides(ten->strides, dims, ndims);

    /* Call operator. */
    fang_ten_ops_arg_t arg = {
        .dest = (fang_gen_t *) ten,

        /* Number of elements. */
        .x = FANG_U2G((fang_uint_t) ten->strides[0] * dims[0]),
        .y = (fang_gen_t) data,
        .z = (fang_gen_t) env
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
    fang_gen_t value)
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
    /* `fang_gen_t` is bitcasted form data types like `fang_float_t` or `fang_int_t`.
       Hence, it can be used as generic representation of all the types. */
    fang_gen_t data[1] = { value };

    fang_ten_ops_arg_t arg = {
        .dest = (fang_gen_t) ten,
        .y = (fang_gen_t) data,
        .z = (fang_gen_t) env
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
int fang_ten_fprint(fang_ten_t *ten, const char *name, int padding,
    FILE *file)
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
        .dest = (fang_gen_t) ten,
        .x = FANG_I2G(padding),
        .y = (fang_gen_t) &buff
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
int fang_ten_rand(fang_ten_t *ten, fang_gen_t low, fang_gen_t high, uint32_t seed) {
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
        .dest = (fang_gen_t) &input,
        .x = low,
        .y = high,
        .z = FANG_U2G(seed)
    };
    res = env->ops->dense->rand(&arg);

out:
    return res;
}

/* Tensor arithmatic macro. Use this macro to instantiate any arithmatic related
   tensor operation routine. */
#define FANG_TENSOR_ARITH(operator)                                             \
int fang_ten_##operator(fang_ten_t *dest, fang_ten_t *x, fang_ten_t *y) {       \
    int res = FANG_OK;                                                          \
                                                                                \
    /* Tensors have to belong to same Environment. */                           \
    if(FANG_UNLIKELY(dest->eid != x->eid || x->eid != y->eid)) {                \
        res = -FANG_ENVNOMATCH;                                                 \
        goto out;                                                               \
    }                                                                           \
                                                                                \
    /* Tensors have to have same data type. */                                  \
    if(FANG_UNLIKELY(dest->dtyp != x->dtyp || x->dtyp != y->dtyp)) {            \
        res = -FANG_INVDTYP;                                                    \
        goto out;                                                               \
    }                                                                           \
                                                                                \
    /* Get Environment. */                                                      \
    fang_env_t *env;                                                            \
    if(FANG_UNLIKELY(!FANG_ISOK(res = _fang_env_retrieve(&env, x->eid))))       \
        goto out;                                                               \
                                                                                \
    /* There are chances of strides and dimension changes during
       broadcasting. */                                                         \
    fang_ten_t wx = *x, wy = *y;                                                \
                                                                                \
    /* Handle scalar tensors. */                                                \
    wx.dims = wx.dims == NULL ? (uint32_t []) { 1 } : wx.dims;                  \
    wy.dims = wy.dims == NULL ? (uint32_t []) { 1 } : wy.dims;                  \
    wx.strides = wx.strides == NULL ? (uint32_t []) { 1 } : wx.strides;         \
    wy.strides = wy.strides == NULL ? (uint32_t []) { 1 } : wy.strides;         \
                                                                                \
    /* Argument structure. */                                                   \
    fang_ten_ops_arg_t arg = {                                                  \
        .dest = (fang_gen_t) dest,                                              \
        .x = (fang_gen_t) &wx,                                                  \
        .y = (fang_gen_t) &wy                                                   \
    };                                                                          \
                                                                                \
    /* Did the tensor got swapped? This information may be relevant for
       operations that are not commutative (e.g. subtraction, division) */      \
    int swapped = 0;                                                            \
                                                                                \
    /* Tensors with same dimension may be batched or broadcasted. */            \
    if(FANG_LIKELY(x->ndims == y->ndims)) {                                     \
        /* Check if tensor operation should be batched. */                      \
        if(FANG_LIKELY(!memcmp(x->dims, y->dims, x->ndims *                     \
            sizeof(*x->dims))))                                                 \
        {                                                                       \
            /* Check if destination tensor is valid to store result. */         \
            if(FANG_UNLIKELY(dest->ndims != x->ndims ||                         \
                memcmp(dest->dims, x->dims, x->ndims * sizeof(*x->dims))))      \
            {                                                                   \
                res = -FANG_DESTINVDIM;                                         \
                goto out;                                                       \
            }                                                                   \
                                                                                \
            /* No need for broadcasting. */                                     \
            arg.z = FANG_I2G(FANG_NO_BCAST);                                    \
            res = env->ops->dense->operator(&arg);                              \
        } else {                                                                \
            /* Check if tensors are broadcastable, if not batched. */           \
            for(int i = 0; i < x->ndims; i++) {                                 \
                uint32_t xdim = x->dims[i];                                     \
                uint32_t ydim = y->dims[i];                                     \
                                                                                \
                /* Not broadcastable. */                                        \
                if(FANG_UNLIKELY(xdim != ydim && xdim != 1 && ydim != 1)) {     \
                    res = -FANG_NOBROAD;                                        \
                    goto out;                                                   \
                }                                                               \
            }                                                                   \
                                                                                \
            /* Check if destination tensor is valid to store result. */         \
            if(FANG_UNLIKELY(dest->ndims != x->ndims)) {                        \
                res = -FANG_DESTINVDIM;                                         \
                goto out;                                                       \
            }                                                                   \
            for(int i = 0; i < x->ndims; i++) {                                 \
                if(FANG_UNLIKELY(dest->dims[i] !=                               \
                    _FANG_MAX(x->dims[i], y->dims[i])))                         \
                {                                                               \
                    res = -FANG_DESTINVDIM;                                     \
                    goto out;                                                   \
                }                                                               \
            }                                                                   \
                                                                                \
            /* Calculate the broadcasted strides. */                            \
            uint32_t x_broadcasted_strides[x->ndims];                           \
            uint32_t y_broadcasted_strides[y->ndims];                           \
            int pattern = _fang_ten_get_broadcast_pattern(&wx, &wy,             \
                &swapped);                                                      \
                                                                                \
            /* No fast route has been found. */                                 \
            if(FANG_UNLIKELY(pattern == FANG_BCAST_UNKNOWN)) {                  \
                /* Pre-compute broadcasted strides. */                          \
                for(int i = 0; i < wx.ndims; i++) {                             \
                    uint32_t xdim = wx.dims[i], ydim = wy.dims[i];              \
                                                                                \
                    x_broadcasted_strides[i] = xdim != 1 ?                      \
                        wx.strides[i] : 0;                                      \
                    y_broadcasted_strides[i] = ydim != 1 ?                      \
                        wy.strides[i] : 0;                                      \
                }                                                               \
                                                                                \
                /* Change strides to broadcasted strides. */                    \
                wx.strides = x_broadcasted_strides;                             \
                wy.strides = y_broadcasted_strides;                             \
            }                                                                   \
                                                                                \
            arg.z = FANG_I2G((swapped << 0x08) | (uint8_t) pattern);            \
            res = env->ops->dense->operator(&arg);                              \
        }                                                                       \
    } else {                                                                    \
        /* Classify tensors based on max/min dimension count. */                \
        int dmax = _FANG_MAX(wx.ndims, wy.ndims),                               \
            dmin = _FANG_MIN(wx.ndims, wy.ndims);                               \
        int diff = dmax - dmin;                                                 \
                                                                                \
        /* Ensure `wx` always holds tensor with maximum dimension count. */     \
        if(FANG_UNLIKELY(wx.ndims < wy.ndims)) {                                \
            fang_ten_t temp = wy;                                               \
            wy = wx;                                                            \
            wx = temp;                                                          \
                                                                                \
            /* Tensors got swapped. */                                          \
            swapped ^= 1;                                                       \
        }                                                                       \
                                                                                \
        /* Check if tensors are broadcastable. */                               \
        for(int i = 0; i < dmin; i++) {                                         \
            uint32_t wxdim = wx.dims[i + diff];                                 \
            uint32_t wydim = wy.dims[i];                                        \
                                                                                \
            /* Not broadcastable. */                                            \
            if(FANG_UNLIKELY(wxdim != wydim && wxdim != 1 && wydim != 1)) {     \
                res = -FANG_NOBROAD;                                            \
                goto out;                                                       \
            }                                                                   \
        }                                                                       \
                                                                                \
        /* Check if destination tensor is valid to store result. */             \
        if(FANG_UNLIKELY(dest->ndims != dmax)) {                                \
            res = -FANG_DESTINVDIM;                                             \
            goto out;                                                           \
        }                                                                       \
        for(int i = 0; i < diff; i++) {                                         \
            if(FANG_UNLIKELY(dest->dims[i] != wx.dims[i])) {                    \
                res = -FANG_DESTINVDIM;                                         \
                goto out;                                                       \
            }                                                                   \
        }                                                                       \
        for(int i = 0; i < dmin; i++) {                                         \
            if(FANG_UNLIKELY(dest->dims[i + diff] !=                            \
                _FANG_MAX(wx.dims[i + diff], wy.dims[i])))                      \
            {                                                                   \
                res = -FANG_DESTINVDIM;                                         \
                goto out;                                                       \
            }                                                                   \
        }                                                                       \
                                                                                \
        int pattern = _fang_ten_get_broadcast_pattern(&wx, &wy, &swapped);      \
        uint32_t x_broadcasted_strides[dmax];                                   \
        uint32_t y_broadcasted_strides[dmax];                                   \
                                                                                \
        /* No fast route has been found. */                                     \
        if(FANG_UNLIKELY(pattern == FANG_BCAST_UNKNOWN)) {                      \
            /* Calculate broadcasted strides. */                                \
            for(int i = 0; i < dmax; i++) {                                     \
                x_broadcasted_strides[i] = wx.dims[i] != 1 ?                    \
                    wx.strides[i] : 0;                                          \
                y_broadcasted_strides[i] = i - diff < 0 ? 0 :                   \
                    (wy.dims[i - diff] != 1 ? wy.strides[i - diff] : 0);        \
            }                                                                   \
                                                                                \
            wx.strides = x_broadcasted_strides;                                 \
            wy.strides = y_broadcasted_strides;                                 \
        }                                                                       \
                                                                                \
        arg.z = FANG_I2G((swapped << 0x08) | (uint8_t) pattern);                \
        res = env->ops->dense->operator(&arg);                                  \
    }                                                                           \
                                                                                \
out:                                                                            \
    return res;                                                                 \
}

/* Adds two tensor. */
FANG_TENSOR_ARITH(sum)

/* Subtracts two tensor. */
FANG_TENSOR_ARITH(diff)

/* Multiplies two tensor. */
FANG_TENSOR_ARITH(mul)

/* Performs General Matrix-Matrix Multiply (GEMM) operation on two trailing
   dimension. */
/* dest := alpha * xy + beta * dest */
int fang_ten_gemm(fang_ten_gemm_transp_t transp_x,
    fang_ten_gemm_transp_t transp_y, fang_gen_t beta, fang_ten_t *dest,
    fang_gen_t alpha, fang_ten_t *x, fang_ten_t *y)
{
    int res = FANG_OK;

    /* Tensors have to belong to same Environment. */
    if(FANG_UNLIKELY(dest->eid != x->eid || x->eid != y->eid)) {
        res = -FANG_ENVNOMATCH;
        goto out;
    }

    /* Tensors have to have same data type. */
    if(FANG_UNLIKELY(dest->dtyp != x->dtyp || x->dtyp != y->dtyp)) {
        res = -FANG_INVDTYP;
        goto out;
    }

    // TODO: Add support for more data types.
    /* Check for supported data types. */
    if(FANG_UNLIKELY(x->dtyp != FANG_TEN_DTYPE_FLOAT32)) {
        res = -FANG_UNSUPDTYP;
        goto out;
    }

    /* Handle scalar tensors. */
    if(FANG_UNLIKELY(x->dims == NULL || y->dims == NULL)) {
        res = -FANG_INVDIM;
        goto out;
    }

    /* Check for GEMM validity. Atleast two dimensional tensor is required and
       commmon dimension should match. */
    if(FANG_UNLIKELY(x->ndims < 2 || y->ndims < 2 ||
        x->dims[x->ndims - 1 - (transp_x == FANG_TEN_GEMM_TRANSPOSE)] !=
        y->dims[y->ndims - 2 + (transp_y == FANG_TEN_GEMM_TRANSPOSE)]))
    {
        res = -FANG_INCMATDIM;
        goto out;
    }

    /* Check if destination tensor's matrix operand can hold the result. */
    if(FANG_UNLIKELY(dest->ndims < 2 ||
        dest->dims[dest->ndims - 2] !=
        x->dims[x->ndims - 2 + (transp_x == FANG_TEN_GEMM_TRANSPOSE)] ||
        dest->dims[dest->ndims - 1] !=
        y->dims[y->ndims - 1 - (transp_y == FANG_TEN_GEMM_TRANSPOSE)]))
    {
        res = -FANG_DESTINVDIM;
        goto out;
    }

    /* The transpose bitmask. */
    /* This would be the passed alongside with fast-route broadcast pattern data
     * in such manner:
     *         | transpose_x bit | transpose_y bit | 8-bit broadcast pattern |
     */
    int transpose_mask = (((transp_x == FANG_TEN_GEMM_TRANSPOSE) << 0x1) |
        (transp_y == FANG_TEN_GEMM_TRANSPOSE)) << 0x08;

    /* Get Environment. */
    fang_env_t *env;
    if(FANG_UNLIKELY(!FANG_ISOK(res = _fang_env_retrieve(&env, x->eid))))
        goto out;

    /* There are chances of strides and dimension changes during
       broadcasting. Also, the tensors might get temporarily commuted. */
    fang_ten_t wx = *x, wy = *y;

    /* Argument structure. */
    fang_ten_ops_arg_t arg = {
        .dest = (fang_gen_t) dest,
        .x = (fang_gen_t) &wx,
        .y = (fang_gen_t) &wy,
        .alpha = alpha,
        .beta = beta
    };

    /* Tensors with same outer dimension count excluding two trailing dimension
     * maybe batched or broadcasted. */
    if(FANG_LIKELY(x->ndims == y->ndims)) {
        /* Check if tensor operation should be batched. */
        if(FANG_LIKELY(!memcmp(x->dims, y->dims,
            (x->ndims - 2) * sizeof(*x->dims))))
        {
            /* Check if destination tensor is valid to store result. */
            if(FANG_UNLIKELY(dest->ndims != x->ndims ||
                memcmp(dest->dims, x->dims, (x->ndims - 2) * sizeof(*x->dims)) ||
                dest->dims[dest->ndims - 1] != y->dims[y->ndims - 1] ||
                dest->dims[dest->ndims - 2] != x->dims[x->ndims - 2]))
            {
                res = -FANG_DESTINVDIM;
                goto out;
            }

            /* No need for broadcasting. */
            arg.z = FANG_I2G(transpose_mask | (uint8_t) FANG_NO_BCAST);
            res = env->ops->dense->gemm(&arg);
        } else {    // Tensor operation cannot be batched
            for(int i = 0; i < x->ndims - 2; i++) {
                uint32_t xdim = wx.dims[i];
                uint32_t ydim = wy.dims[i];

                /* Not broadcastable. */
                if(FANG_UNLIKELY(xdim != ydim && xdim != 1 && ydim != 1)) {
                    res = -FANG_NOBROAD;
                    goto out;
                }
            }

            /* Check if destination tensor is valid to store result. */
            if(FANG_UNLIKELY(dest->ndims != x->ndims ||
                dest->dims[dest->ndims - 1] != y->dims[y->ndims - 1] ||
                dest->dims[dest->ndims - 2] != x->dims[x->ndims - 2]))
            {
                res = -FANG_DESTINVDIM;
                goto out;
            }
            for(int i = 0; i < x->ndims - 2; i++) {
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
            int pattern = _fang_ten_gemm_get_broadcast_pattern(&wx, &wy);

            /* No fast route has been found. */
            if(FANG_UNLIKELY(pattern == FANG_BCAST_UNKNOWN)) {
                /* Pre-compute broadcasted strides. */
                for(int i = 0; i < wx.ndims - 2; i++) {
                    uint32_t xdim = wx.dims[i], ydim = wy.dims[i];

                    x_broadcasted_strides[i] = xdim != 1 ? wx.strides[i] : 0;
                    y_broadcasted_strides[i] = ydim != 1 ? wy.strides[i] : 0;
                }

                /* Change strides to broadcasted strides. */
                wx.strides = x_broadcasted_strides;
                wy.strides = y_broadcasted_strides;
            }

            arg.z = FANG_I2G(transpose_mask | (uint8_t) pattern);
            res = env->ops->dense->gemm(&arg);
        }
    } else {
        /* Classify tensors based on max/min dimension count. */
        int dmax = _FANG_MAX(wx.ndims, wy.ndims),
            dmin = _FANG_MIN(wx.ndims, wy.ndims);
        int diff = dmax - dmin;

        /* Ensure `wx` always holds tensor with maximum dimension count. */
        if(FANG_UNLIKELY(wx.ndims < wy.ndims)) {
            fang_ten_t temp = wy;
            wy = wx;
            wx = temp;
        }

        /* Check if tensors are broadcastable. */
        for(int i = 0; i < dmin - 2; i++) {  // Exclude the matrix dimensions
            uint32_t wxdim = wx.dims[i + diff];
            uint32_t wydim = wy.dims[i];

            /* Not broadcastable. */
            if(FANG_UNLIKELY(wxdim != wydim && wxdim != 1 && wydim != 1)) {
                res = -FANG_NOBROAD;
                goto out;
            }
        }

        /* Check if destination tensor is valid to store result. */
        if(FANG_UNLIKELY(dest->ndims != dmax)) {
            res = -FANG_DESTINVDIM;
            goto out;
        }
        for(int i = 0; i < diff; i++) {
            if(FANG_UNLIKELY(dest->dims[i] != wx.dims[i])) {
                res = -FANG_DESTINVDIM;
                goto out;
            }
        }
        for(int i = 0; i < dmin - 2; i++) {  // Adjust excluding matrix shape
            if(FANG_UNLIKELY(dest->dims[i + diff] !=
                _FANG_MAX(wx.dims[i + diff], wy.dims[i])))
            {
                res = -FANG_DESTINVDIM;
                goto out;
            }
        }

        int pattern = _fang_ten_gemm_get_broadcast_pattern(&wx, &wy);
        uint32_t x_broadcasted_strides[dmax];
        uint32_t y_broadcasted_strides[dmax];

        /* No fast route has been found. */
        if(FANG_UNLIKELY(pattern == FANG_BCAST_UNKNOWN)) {
            /* Calculate broadcasted strides. */
            for(int i = 0; i < dmax - 2; i++) {
                x_broadcasted_strides[i] = wx.dims[i] != 1 ? wx.strides[i] : 0;
                y_broadcasted_strides[i] = i - diff < 0 ? 0 :
                    (wy.dims[i - diff] != 1 ? wy.strides[i - diff] : 0);
            }

            wx.strides = x_broadcasted_strides;
            wy.strides = y_broadcasted_strides;
        }

        arg.z = FANG_I2G(transpose_mask | (uint8_t) pattern);
        res = env->ops->dense->gemm(&arg);
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
        .dest = (fang_gen_t) ten,
        .z = (fang_gen_t) env
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
