/* ================ MACROS ================ */

/* Makes life easier. */
/* NOTE: Order conforms to `fang_ten_dtype_t` enum. */
#define _ACCEL_DENSE(name)         \
_fang_dense_accel_##name##i8,      \
_fang_dense_accel_##name##i16,     \
_fang_dense_accel_##name##i32,     \
_fang_dense_accel_##name##i64,     \
_fang_dense_accel_##name##u8,      \
_fang_dense_accel_##name##u16,     \
_fang_dense_accel_##name##u32,     \
_fang_dense_accel_##name##u64,     \
_fang_dense_accel_##name##f8,      \
_fang_dense_accel_##name##f16,     \
_fang_dense_accel_##name##bf16,    \
_fang_dense_accel_##name##f32,     \
_fang_dense_accel_##name##f64,

/* ================ MACROS END ================ */


/* ================ PRIVATE DEFINITIONS ================ */

/* Get non-broadcasted linear data index of input tensors w.r.t.
   output tensor. */
FANG_HOT FANG_INLINE static inline void
_fang_get_original_idx(int idx, int *restrict idx_x, int *restrict
    idx_y, uint32_t *dest_strides, uint32_t *x_strides, uint32_t *y_strides,
    int ndims)
{
    *idx_x = *idx_y = 0;
    for(int i = 0; i < ndims; i++) {
        uint32_t dim_idx = idx / dest_strides[i];
        /* The stride here is expected to be pre-broadcasted. */
        *idx_x += dim_idx * x_strides[i];
        *idx_y += dim_idx * y_strides[i];
        idx %= dest_strides[i];
    }
}

/* ================ PRIVATE DEFINITIONS END ================ */


/* ================ ACCELERATOR FUNCTIONS ================ */

/* Common. */
#define _ACCEL_RAND_PROLOGUE(type, prefix)                         \
    fang_ten_t *ten = (fang_ten_t *) arg->dest;  /* Tensor */      \
    type diff     = (type) FANG_G2##prefix(arg->x);  /* Diff */    \
    type low      = (type) FANG_G2##prefix(arg->y);  /* Low */     \
    int size      = (int) ten->strides[0] * ten->dims[0];          \
    uint32_t seed = (uint32_t) FANG_G2U(arg->z);

/* Common for integer types. */
#define _ACCEL_RANDI(prefix, postfix, type, annot)                       \
FANG_HOT FANG_FLATTEN static void                                        \
_fang_dense_accel_rand##postfix(_fang_cpu_accel_arg_t *restrict arg) {   \
    _ACCEL_RAND_PROLOGUE(type, prefix);                                  \
    type *data = (type *) ten->data.dense;                               \
    for(int i = 0; i < size; i++) {                                      \
        data[i] = (rand_r(&seed) &                                       \
            annot##_MAX) % diff + low;                                   \
    }                                                                    \
}

/* Common for float types. */
#define _ACCEL_RANDF(bits, type)                                        \
FANG_HOT FANG_FLATTEN static void                                       \
_fang_dense_accel_randf##bits(_fang_cpu_accel_arg_t *restrict arg) {    \
    _ACCEL_RAND_PROLOGUE(type, F);                                      \
    type *data = (type *) ten->data.dense;                              \
    for(int i = 0; i < size; i++) {                                     \
        data[i] = ((type) rand_r(&seed) /                               \
            RAND_MAX) * diff + low;                                     \
    }                                                                   \
}

/* Integer type. */
_ACCEL_RANDI(I, i8, int8_t, INT8)
_ACCEL_RANDI(I, i16, int16_t, INT16)
_ACCEL_RANDI(I, i32, int32_t, INT32)
_ACCEL_RANDI(I, i64, int64_t, INT64)
_ACCEL_RANDI(U, u8, uint8_t, UINT8)
_ACCEL_RANDI(U, u16, uint16_t, UINT16)
_ACCEL_RANDI(U, u32, uint32_t, UINT32)
_ACCEL_RANDI(U, u64, uint64_t, UINT64)

FANG_HOT FANG_FLATTEN void
    _fang_dense_accel_randf8(_fang_cpu_accel_arg_t *restrict arg)
{
    _ACCEL_RAND_PROLOGUE(float, F);
    _fang_float8_t *data = (_fang_float8_t *) ten->data.dense;

    for(int i = 0; i < size; i++) {
        data[i] = _FANG_S2Q(((float) rand_r(&seed) / RAND_MAX) * diff + low);
    }
}

FANG_HOT FANG_FLATTEN void
    _fang_dense_accel_randf16(_fang_cpu_accel_arg_t *restrict arg)
{
    _ACCEL_RAND_PROLOGUE(float, F);
    _fang_float16_t *data = (_fang_float16_t *) ten->data.dense;

    for(int i = 0; i < size; i++) {
        data[i] = _FANG_S2H(((float) rand_r(&seed) / RAND_MAX) * diff + low);
    }
}

FANG_HOT FANG_FLATTEN void
    _fang_dense_accel_randbf16(_fang_cpu_accel_arg_t *restrict arg)
{
    _ACCEL_RAND_PROLOGUE(float, F);
    _fang_bfloat16_t *data = (_fang_bfloat16_t *) ten->data.dense;

    for(int i = 0; i < size; i++) {
        data[i] = _FANG_S2BH(((float) rand_r(&seed) /
            RAND_MAX) * diff + low);
    }
}

_ACCEL_RANDF(32, float)
_ACCEL_RANDF(64, double)

/* ======== RAND END ======== */

/* ======== SUM ======== */

/* Common. */
#define _ACCEL_SUM_PROLOGUE(type)                                 \
    fang_ten_t *dest = (fang_ten_t *) arg->dest;                  \
    fang_ten_t *x    = (fang_ten_t *) arg->x;                     \
    fang_ten_t *y    = (fang_ten_t *) arg->y;                     \
    type *data_dest  = (type *) dest->data.dense;                 \
    type *data_x     = (type *) x->data.dense;                    \
    type *data_y     = (type *) y->data.dense;                    \
    int size         = (int) dest->strides[0] * dest->dims[0];    \
    int broadcast    = (int) FANG_G2I(arg->z);                    \
    int vsiz         = 0;                                         \
    if(FANG_LIKELY(broadcast == 2))  /* Vector. */                \
        vsiz = y->dims[y->ndims - 1];                             \
    else if(FANG_LIKELY(broadcast == 3))  /* Matrix. */           \
        vsiz = y->dims[0] * y->dims[1];

/* Integer types. */
FANG_HOT FANG_FLATTEN static void
    _fang_dense_accel_sumi8(_fang_cpu_accel_arg_t *restrict arg)
{
    _ACCEL_SUM_PROLOGUE(int8_t);

    /* Scalar tensor addition against N-dimensional tensor. */
    if(FANG_LIKELY(broadcast == 1)) {
        for(int i = 0; i < size; i++)
            data_dest[i] = data_x[i] + data_y[0];
    }
    /* Vector addition against N-dimensional tensor. */
    else if(FANG_LIKELY(broadcast == 2 || broadcast == 3)) {
        /* Another way to go would be to use modulus operation, which is much
           more computation intensive than just using nested loops. */
        for(int i = 0; i < size; i += vsiz) {
            for(int j = 0; j < vsiz; j++)
                data_dest[i + j] = data_x[i + j] + data_y[j];
        }
    }
    /* Broadcast dimension unknown. */
    else if(FANG_UNLIKELY(broadcast == 4)) {
        for(int i = 0; i < size; i++) {
            int idx_x, idx_y;
            /* Get broadcasted flattened index for x and y. */
            _fang_get_original_idx(i, &idx_x, &idx_y, dest->strides,
                x->strides, y->strides, x->ndims);

            data_dest[i] = data_x[idx_x] + data_y[idx_y];
        }
    } else {
        for(int i = 0; i < size; i++)
            data_dest[i] = data_x[i] + data_y[i];
    }
}

FANG_HOT FANG_FLATTEN void
    _fang_dense_accel_sumi16(_fang_cpu_accel_arg_t *restrict arg)
{
    _ACCEL_SUM_PROLOGUE(int16_t);

    /* Scalar tensor addition against N-dimensional tensor. */
    if(FANG_LIKELY(broadcast == 1)) {
        for(int i = 0; i < size; i++)
            data_dest[i] = data_x[i] + data_y[0];
    }
    /* Vector addition against N-dimensional tensor. */
    else if(FANG_LIKELY(broadcast == 2 || broadcast == 3)) {
        /* Another way to go would be to use modulus operation, which is much
           more computation intensive than just using nested loops. */
        for(int i = 0; i < size; i += vsiz) {
            for(int j = 0; j < vsiz; j++)
                data_dest[i + j] = data_x[i + j] + data_y[j];
        }
    }
    /* Broadcast dimension unknown. */
    else if(FANG_UNLIKELY(broadcast == 4)) {
        for(int i = 0; i < size; i++) {
            int idx_x, idx_y;
            /* Get broadcasted flattened index for x and y. */
            _fang_get_original_idx(i, &idx_x, &idx_y, dest->strides,
                x->strides, y->strides, x->ndims);

            data_dest[i] = data_x[idx_x] + data_y[idx_y];
        }
    } else {
        for(int i = 0; i < size; i++)
            data_dest[i] = data_x[i] + data_y[i];
    }
}

FANG_HOT FANG_FLATTEN void
    _fang_dense_accel_sumi32(_fang_cpu_accel_arg_t *restrict arg)
{
    _ACCEL_SUM_PROLOGUE(int32_t);

    /* Scalar tensor addition against N-dimensional tensor. */
    if(FANG_LIKELY(broadcast == 1)) {
        for(int i = 0; i < size; i++)
            data_dest[i] = data_x[i] + data_y[0];
    }
    /* Vector addition against N-dimensional tensor. */
    else if(FANG_LIKELY(broadcast == 2 || broadcast == 3)) {
        /* Another way to go would be to use modulus operation, which is much
           more computation intensive than just using nested loops. */
        for(int i = 0; i < size; i += vsiz) {
            for(int j = 0; j < vsiz; j++)
                data_dest[i + j] = data_x[i + j] + data_y[j];
        }
    }
    /* Broadcast dimension unknown. */
    else if(FANG_UNLIKELY(broadcast == 4)) {
        for(int i = 0; i < size; i++) {
            int idx_x, idx_y;
            /* Get broadcasted flattened index for x and y. */
            _fang_get_original_idx(i, &idx_x, &idx_y, dest->strides,
                x->strides, y->strides, x->ndims);

            data_dest[i] = data_x[idx_x] + data_y[idx_y];
        }
    } else {
        for(int i = 0; i < size; i++)
            data_dest[i] = data_x[i] + data_y[i];
    }
}

FANG_HOT FANG_FLATTEN void
    _fang_dense_accel_sumi64(_fang_cpu_accel_arg_t *restrict arg)
{
    _ACCEL_SUM_PROLOGUE(int64_t);

    /* Scalar tensor addition against N-dimensional tensor. */
    if(FANG_LIKELY(broadcast == 1)) {
        for(int i = 0; i < size; i++)
            data_dest[i] = data_x[i] + data_y[0];
    }
    /* Vector addition against N-dimensional tensor. */
    else if(FANG_LIKELY(broadcast == 2 || broadcast == 3)) {
        /* Another way to go would be to use modulus operation, which is much
           more computation intensive than just using nested loops. */
        for(int i = 0; i < size; i += vsiz) {
            for(int j = 0; j < vsiz; j++)
                data_dest[i + j] = data_x[i + j] + data_y[j];
        }
    }
    /* Broadcast dimension unknown. */
    else if(FANG_UNLIKELY(broadcast == 4)) {
        for(int i = 0; i < size; i++) {
            int idx_x, idx_y;
            /* Get broadcasted flattened index for x and y. */
            _fang_get_original_idx(i, &idx_x, &idx_y, dest->strides,
                x->strides, y->strides, x->ndims);

            data_dest[i] = data_x[idx_x] + data_y[idx_y];
        }
    } else {
        for(int i = 0; i < size; i++)
            data_dest[i] = data_x[i] + data_y[i];
    }
}

FANG_HOT FANG_FLATTEN void
    _fang_dense_accel_sumu8(_fang_cpu_accel_arg_t *restrict arg)
{
    _ACCEL_SUM_PROLOGUE(uint8_t);

    /* Scalar tensor addition against N-dimensional tensor. */
    if(FANG_LIKELY(broadcast == 1)) {
        for(int i = 0; i < size; i++)
            data_dest[i] = data_x[i] + data_y[0];
    }
    /* Vector addition against N-dimensional tensor. */
    else if(FANG_LIKELY(broadcast == 2 || broadcast == 3)) {
        /* Another way to go would be to use modulus operation, which is much
           more computation intensive than just using nested loops. */
        for(int i = 0; i < size; i += vsiz) {
            for(int j = 0; j < vsiz; j++)
                data_dest[i + j] = data_x[i + j] + data_y[j];
        }
    }
    /* Broadcast dimension unknown. */
    else if(FANG_UNLIKELY(broadcast == 4)) {
        for(int i = 0; i < size; i++) {
            int idx_x, idx_y;
            /* Get broadcasted flattened index for x and y. */
            _fang_get_original_idx(i, &idx_x, &idx_y, dest->strides,
                x->strides, y->strides, x->ndims);

            data_dest[i] = data_x[idx_x] + data_y[idx_y];
        }
    } else {
        for(int i = 0; i < size; i++)
            data_dest[i] = data_x[i] + data_y[i];
    }
}

FANG_HOT FANG_FLATTEN void
    _fang_dense_accel_sumu16(_fang_cpu_accel_arg_t *restrict arg)
{
    _ACCEL_SUM_PROLOGUE(uint16_t);

    /* Scalar tensor addition against N-dimensional tensor. */
    if(FANG_LIKELY(broadcast == 1)) {
        for(int i = 0; i < size; i++)
            data_dest[i] = data_x[i] + data_y[0];
    }
    /* Vector addition against N-dimensional tensor. */
    else if(FANG_LIKELY(broadcast == 2 || broadcast == 3)) {
        /* Another way to go would be to use modulus operation, which is much
           more computation intensive than just using nested loops. */
        for(int i = 0; i < size; i += vsiz) {
            for(int j = 0; j < vsiz; j++)
                data_dest[i + j] = data_x[i + j] + data_y[j];
        }
    }
    /* Broadcast dimension unknown. */
    else if(FANG_UNLIKELY(broadcast == 4)) {
        for(int i = 0; i < size; i++) {
            int idx_x, idx_y;

            /* Get broadcasted flattened index for x and y. */
            _fang_get_original_idx(i, &idx_x, &idx_y, dest->strides,
                x->strides, y->strides, x->ndims);

            data_dest[i] = data_x[idx_x] + data_y[idx_y];
        }
    } else {
        for(int i = 0; i < size; i++)
            data_dest[i] = data_x[i] + data_y[i];
    }
}

FANG_HOT FANG_FLATTEN void
    _fang_dense_accel_sumu32(_fang_cpu_accel_arg_t *restrict arg)
{
    _ACCEL_SUM_PROLOGUE(uint32_t);

    /* Scalar tensor addition against N-dimensional tensor. */
    if(FANG_LIKELY(broadcast == 1)) {
        for(int i = 0; i < size; i++)
            data_dest[i] = data_x[i] + data_y[0];
    }
    /* Vector addition against N-dimensional tensor. */
    else if(FANG_LIKELY(broadcast == 2 || broadcast == 3)) {
        /* Another way to go would be to use modulus operation, which is much
           more computation intensive than just using nested loops. */
        for(int i = 0; i < size; i += vsiz) {
            for(int j = 0; j < vsiz; j++)
                data_dest[i + j] = data_x[i + j] + data_y[j];
        }
    }
    /* Broadcast dimension unknown. */
    else if(FANG_UNLIKELY(broadcast == 4)) {
        for(int i = 0; i < size; i++) {
            int idx_x, idx_y;
            /* Get broadcasted flattened index for x and y. */
            _fang_get_original_idx(i, &idx_x, &idx_y, dest->strides,
                x->strides, y->strides, x->ndims);

            data_dest[i] = data_x[idx_x] + data_y[idx_y];
        }
    } else {
        for(int i = 0; i < size; i++)
            data_dest[i] = data_x[i] + data_y[i];
    }
}

FANG_HOT FANG_FLATTEN void
    _fang_dense_accel_sumu64(_fang_cpu_accel_arg_t *restrict arg)
{
    _ACCEL_SUM_PROLOGUE(uint64_t);

    /* Scalar tensor addition against N-dimensional tensor. */
    if(FANG_LIKELY(broadcast == 1)) {
        for(int i = 0; i < size; i++)
            data_dest[i] = data_x[i] + data_y[0];
    }
    /* Vector addition against N-dimensional tensor. */
    else if(FANG_LIKELY(broadcast == 2 || broadcast == 3)) {
        /* Another way to go would be to use modulus operation, which is much
           more computation intensive than just using nested loops. */
        for(int i = 0; i < size; i += vsiz) {
            for(int j = 0; j < vsiz; j++)
                data_dest[i + j] = data_x[i + j] + data_y[j];
        }
    }
    /* Broadcast dimension unknown. */
    else if(FANG_UNLIKELY(broadcast == 4)) {
        for(int i = 0; i < size; i++) {
            int idx_x, idx_y;
            /* Get broadcasted flattened index for x and y. */
            _fang_get_original_idx(i, &idx_x, &idx_y, dest->strides,
                x->strides, y->strides, x->ndims);

            data_dest[i] = data_x[idx_x] + data_y[idx_y];
        }
    } else {
        for(int i = 0; i < size; i++)
            data_dest[i] = data_x[i] + data_y[i];
    }
}

/* Float types. */
FANG_HOT FANG_FLATTEN void
    _fang_dense_accel_sumf8(_fang_cpu_accel_arg_t *restrict arg)
{
    _ACCEL_SUM_PROLOGUE(_fang_float8_t);

    /* Scalar tensor addition against N-dimensional tensor. */
    if(FANG_LIKELY(broadcast == 1)) {
        for(int i = 0; i < size; i++) {
            data_dest[i] = _FANG_S2Q(_FANG_Q2S(data_x[i]) +
                _FANG_Q2S(data_y[0]));
        }
    }
    /* Vector addition against N-dimensional tensor. */
    else if(FANG_LIKELY(broadcast == 2 || broadcast == 3)) {
        /* Another way to go would be to use modulus operation, which is much
           more computation intensive than just using nested loops. */
        for(int i = 0; i < size; i += vsiz) {
            for(int j = 0; j < vsiz; j++) {
                data_dest[i + j] = _FANG_S2Q(_FANG_Q2S(data_x[i + j]) +
                    _FANG_Q2S(data_y[j]));
            }
        }
    }
    /* Broadcast dimension unknown. */
    else if(FANG_UNLIKELY(broadcast == 4)) {
        for(int i = 0; i < size; i++) {
            int idx_x, idx_y;
            /* Get broadcasted flattened index for x and y. */
            _fang_get_original_idx(i, &idx_x, &idx_y, dest->strides,
                x->strides, y->strides, x->ndims);

            data_dest[i] = _FANG_S2Q(_FANG_Q2S(data_x[idx_x]) +
                _FANG_Q2S(data_y[idx_y]));
        }
    } else {
        for(int i = 0; i < size; i++) {
            data_dest[i] = _FANG_S2Q(_FANG_Q2S(data_x[i]) +
                _FANG_Q2S(data_y[i]));
        }
    }
}

FANG_HOT FANG_FLATTEN void
    _fang_dense_accel_sumf16(_fang_cpu_accel_arg_t *restrict arg)
{
    _ACCEL_SUM_PROLOGUE(_fang_float16_t);

    /* Scalar tensor addition against N-dimensional tensor. */
    if(FANG_LIKELY(broadcast == 1)) {
        for(int i = 0; i < size; i++) {
            data_dest[i] = _FANG_S2H(_FANG_H2S(data_x[i]) +
                _FANG_H2S(data_y[0]));
        }
    }
    /* Vector addition against N-dimensional tensor. */
    else if(FANG_LIKELY(broadcast == 2 || broadcast == 3)) {
        /* Another way to go would be to use modulus operation, which is much
           more computation intensive than just using nested loops. */
        for(int i = 0; i < size; i += vsiz) {
            for(int j = 0; j < vsiz; j++) {
                data_dest[i + j] = _FANG_S2H(_FANG_H2S(data_x[i + j]) +
                    _FANG_H2S(data_y[j]));
            }
        }
    }
    /* Broadcast dimension unknown. */
    else if(FANG_UNLIKELY(broadcast == 4)) {
        for(int i = 0; i < size; i++) {
            int idx_x, idx_y;
            /* Get broadcasted flattened index for x and y. */
            _fang_get_original_idx(i, &idx_x, &idx_y, dest->strides,
                x->strides, y->strides, x->ndims);

            data_dest[i] = _FANG_S2H(_FANG_H2S(data_x[idx_x]) +
                _FANG_H2S(data_y[idx_y]));
        }
    } else {
        for(int i = 0; i < size; i++) {
            data_dest[i] = _FANG_S2H(_FANG_H2S(data_x[i]) +
                _FANG_H2S(data_y[i]));
        }
    }
}

FANG_HOT FANG_FLATTEN void
    _fang_dense_accel_sumbf16(_fang_cpu_accel_arg_t *restrict arg)
{
    _ACCEL_SUM_PROLOGUE(_fang_bfloat16_t);

    /* Scalar tensor addition against N-dimensional tensor. */
    if(FANG_LIKELY(broadcast == 1)) {
        for(int i = 0; i < size; i++) {
            data_dest[i] = _FANG_S2BH(_FANG_BH2S(data_x[i]) +
                _FANG_BH2S(data_y[0]));
        }
    }
    /* Vector addition against N-dimensional tensor. */
    else if(FANG_LIKELY(broadcast == 2 || broadcast == 3)) {
        /* Another way to go would be to use modulus operation, which is much
           more computation intensive than just using nested loops. */
        for(int i = 0; i < size; i += vsiz) {
            for(int j = 0; j < vsiz; j++) {
                data_dest[i + j] = _FANG_S2BH(_FANG_BH2S(data_x[i + j]) +
                    _FANG_BH2S(data_y[j]));
            }
        }
    }
    /* Broadcast dimension unknown. */
    else if(FANG_UNLIKELY(broadcast == 4)) {
        for(int i = 0; i < size; i++) {
            int idx_x, idx_y;
            /* Get broadcasted flattened index for x and y. */
            _fang_get_original_idx(i, &idx_x, &idx_y, dest->strides,
                x->strides, y->strides, x->ndims);

            data_dest[i] = _FANG_S2BH(_FANG_BH2S(data_x[idx_x]) +
                _FANG_BH2S(data_y[idx_y]));
        }
    } else {
        for(int i = 0; i < size; i++) {
            data_dest[i] = _FANG_S2BH(_FANG_BH2S(data_x[i]) +
                _FANG_BH2S(data_y[i]));
        }
    }
}

FANG_HOT FANG_FLATTEN void
    _fang_dense_accel_sumf32(_fang_cpu_accel_arg_t *restrict arg)
{
    _ACCEL_SUM_PROLOGUE(float);

    /* Scalar tensor addition against N-dimensional tensor. */
    if(FANG_LIKELY(broadcast == 1)) {
        for(int i = 0; i < size; i++)
            data_dest[i] = data_x[i] + data_y[0];
    }
    /* Vector addition against N-dimensional tensor. */
    else if(FANG_LIKELY(broadcast == 2 || broadcast == 3)) {
        /* Another way to go would be to use modulus operation, which is much
           more computation intensive than just using nested loops. */
        for(int i = 0; i < size; i += vsiz) {
            for(int j = 0; j < vsiz; j++)
                data_dest[i + j] = data_x[i + j] + data_y[j];
        }
    }
    /* Broadcast dimension unknown. */
    else if(FANG_UNLIKELY(broadcast == 4)) {
        for(int i = 0; i < size; i++) {
            int idx_x, idx_y;
            /* Get broadcasted flattened index for x and y. */
            _fang_get_original_idx(i, &idx_x, &idx_y, dest->strides,
                x->strides, y->strides, x->ndims);

            data_dest[i] = data_x[idx_x] + data_y[idx_y];
        }
    } else {
        for(int i = 0; i < size; i++)
            data_dest[i] = data_x[i] + data_y[i];
    }
}

FANG_HOT FANG_FLATTEN void
    _fang_dense_accel_sumf64(_fang_cpu_accel_arg_t *restrict arg)
{
    _ACCEL_SUM_PROLOGUE(double);

    /* Scalar tensor addition against N-dimensional tensor. */
    if(FANG_LIKELY(broadcast == 1)) {
        for(int i = 0; i < size; i++)
            data_dest[i] = data_x[i] + data_y[0];
    }
    /* Vector addition against N-dimensional tensor. */
    else if(FANG_LIKELY(broadcast == 2 || broadcast == 3)) {
        /* Another way to go would be to use modulus operation, which is much
           more computation intensive than just using nested loops. */
        for(int i = 0; i < size; i += vsiz) {
            for(int j = 0; j < vsiz; j++)
                data_dest[i + j] = data_x[i + j] + data_y[j];
        }
    }
    /* Broadcast dimension unknown. */
    else if(FANG_UNLIKELY(broadcast == 4)) {
        for(int i = 0; i < size; i++) {
            int idx_x, idx_y;
            /* Get broadcasted flattened index for x and y. */
            _fang_get_original_idx(i, &idx_x, &idx_y, dest->strides,
                x->strides, y->strides, x->ndims);

            data_dest[i] = data_x[idx_x] + data_y[idx_y];
        }
    } else {
        for(int i = 0; i < size; i++)
            data_dest[i] = data_x[i] + data_y[i];
    }
}

#undef _ACCEL_SUM_PROLOGUE

/* ======== SUM END ======== */

/* ================ ACCELERATOR FUNCTIONS END ================ */

