/* ================ MACROS ================ */

/* Makes life easier. */
/* NOTE: Order conforms to `fang_ten_dtype_t` enum. */
/* In lower levels, integers don't have signedness in operations. Hence,
   operating on signed or unsigned integers are the same thing. */
#define _ACCEL_DENSE(name)         \
_fang_dense_accel_##name##i8,      \
_fang_dense_accel_##name##i16,     \
_fang_dense_accel_##name##i32,     \
_fang_dense_accel_##name##i64,     \
_fang_dense_accel_##name##i8,      \
_fang_dense_accel_##name##i16,     \
_fang_dense_accel_##name##i32,     \
_fang_dense_accel_##name##i64,     \
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
#define _ACCEL_RAND_PROLOGUE(type, prefix)                              \
    fang_ten_t *ten = (fang_ten_t *) arg->dest;  /* Tensor */           \
    type diff     = (type) FANG_G2##prefix(arg->x);  /* Diff */         \
    type low      = (type) FANG_G2##prefix(arg->y);  /* Low */          \
    int size      = ten->dims == NULL ? 1 :                             \
        (int) ten->strides[0] * ten->dims[0];                           \
    uint32_t seed = (uint32_t) FANG_G2U(arg->z);

/* Common for integer types. */
#define _ACCEL_RANDI(bits, type, annot)                                 \
FANG_HOT FANG_FLATTEN static void                                       \
_fang_dense_accel_randi##bits(_fang_cpu_accel_arg_t *restrict arg) {    \
    _ACCEL_RAND_PROLOGUE(type, I);                                      \
    type *data = (type *) ten->data.dense;                              \
    for(int i = 0; i < size; i++) {                                     \
        data[i] = (rand_r(&seed) &                                      \
            annot##_MAX) % diff + low;                                  \
    }                                                                   \
}

/* Common for float types. */
#define _ACCEL_RANDF(dt, type, type_cast, conv_a2b)                     \
FANG_HOT FANG_FLATTEN static void                                       \
_fang_dense_accel_rand##dt(_fang_cpu_accel_arg_t *restrict arg) {       \
    _ACCEL_RAND_PROLOGUE(type_cast, F);                                 \
    type *data = (type *) ten->data.dense;                              \
    for(int i = 0; i < size; i++) {                                     \
        data[i] = conv_a2b(((type_cast) rand_r(&seed) /                 \
            RAND_MAX) * diff + low);                                    \
    }                                                                   \
}

/* Integer type. */
_ACCEL_RANDI(8, int8_t, INT8)
_ACCEL_RANDI(16, int16_t, INT16)
_ACCEL_RANDI(32, int32_t, INT32)
_ACCEL_RANDI(64, int64_t, INT64)

/* Floating point types. */
_ACCEL_RANDF(f8, _fang_float8_t, float, _FANG_S2Q)
_ACCEL_RANDF(f16, _fang_float16_t, float, _FANG_S2H)
_ACCEL_RANDF(bf16, _fang_bfloat16_t, float, _FANG_S2BH)
_ACCEL_RANDF(f32, float, float,)
_ACCEL_RANDF(f64, double, double,)

/* ======== RAND END ======== */

/* ======== ARITHMATIC ACCELERATOR HELPERS ======== */

/* Common. */
#define _ACCEL_ARITH_PROLOGUE(type)                                            \
    fang_ten_t *dest = (fang_ten_t *) arg->dest;                               \
    fang_ten_t *x    = (fang_ten_t *) arg->x;                                  \
    fang_ten_t *y    = (fang_ten_t *) arg->y;                                  \
    type *data_dest  = FANG_ASSUME_ALIGNED(dest->data.dense, 64);              \
    type *data_x     = FANG_ASSUME_ALIGNED(x->data.dense, 64);                 \
    type *data_y     = FANG_ASSUME_ALIGNED(y->data.dense, 64);                 \
    int size         = dest->dims == NULL ? 1 :                                \
        (int) dest->strides[0] * dest->dims[0];                                \
    int swapb_mask   = (int) FANG_G2I(arg->z);                                 \
    int broadcast    = swapb_mask & 0xFF;                                      \
    int vsiz         = 0;  /* Vector size */                                   \
    if(FANG_LIKELY(broadcast == FANG_BCAST_ROWVEC))                            \
        vsiz = y->dims[y->ndims - 1];                                          \
    else if(FANG_LIKELY(broadcast == FANG_BCAST_COLVEC))                       \
        vsiz = y->dims[y->ndims - 2];                                          \
    else if(FANG_LIKELY(broadcast == FANG_BCAST_MATRIX))                       \
        vsiz = y->dims[y->ndims - 2] * y->dims[y->ndims - 2];

/* Mostly common for all types of arithmatic. */
#define _ACCEL_ARITHMATIC(postfix, type, op, conv_a2b, conv_b2a)               \
FANG_HOT FANG_FLATTEN static void                                              \
    _fang_dense_accel_##postfix(_fang_cpu_accel_arg_t *restrict arg)           \
{                                                                              \
    _ACCEL_ARITH_PROLOGUE(type);                                               \
                                                                               \
    /* Scalar tensor operation against N-dimensional tensor. */                \
    if(FANG_LIKELY(broadcast == FANG_BCAST_SCALAR)) {                          \
        for(int i = 0; i < size; i++)                                          \
            data_dest[i] = conv_b2a(conv_a2b(data_x[i]) op                     \
                conv_a2b(data_y[0]));                                          \
    }                                                                          \
    /* Row-major vector/matrix operation against N-dimensional tensor. */      \
    else if(FANG_LIKELY(broadcast == FANG_BCAST_ROWVEC ||                      \
        broadcast == FANG_BCAST_MATRIX))                                       \
    {                                                                          \
        /* Another way to go would be to use modulus operation, which is
           much more computation intensive than just using nested loops. */    \
        for(int i = 0; i < size; i += vsiz) {                                  \
            for(int j = 0; j < vsiz; j++)                                      \
                data_dest[i + j] = conv_b2a(conv_a2b(data_x[i + j]) op         \
                    conv_a2b(data_y[j]));                                      \
        }                                                                      \
    }                                                                          \
    /* Col-major vector operation against N-dimensional tensor. */             \
    else if(FANG_LIKELY(broadcast == FANG_BCAST_COLVEC)) {                     \
        int xvsiz = x->dims[x->ndims - 1];                                     \
        int skip = xvsiz * vsiz;                                               \
                                                                               \
        for(int i = 0; i < size; i += skip) {                                  \
            for(int j = 0, idx = 0; j < skip; j += xvsiz, idx++) {             \
                for(int k = 0; k < xvsiz; k++)                                 \
                    data_dest[i + j + k] =                                     \
                        conv_b2a(conv_a2b(data_x[i + j + k]) op                \
                        conv_a2b(data_y[idx]));                                \
            }                                                                  \
        }                                                                      \
    }                                                                          \
    /* Broadcast dimension unknown. */                                         \
    else if(FANG_UNLIKELY(broadcast == FANG_BCAST_UNKNOWN)) {                  \
        for(int i = 0; i < size; i++) {                                        \
            int idx_x, idx_y;                                                  \
            /* Get broadcasted flattened index for x and y. */                 \
            _fang_get_original_idx(i, &idx_x, &idx_y, dest->strides,           \
                x->strides, y->strides, x->ndims);                             \
            data_dest[i] = conv_b2a(conv_a2b(data_x[idx_x]) op                 \
                conv_a2b(data_y[idx_y]));                                      \
        }                                                                      \
    } else {                                                                   \
        for(int i = 0; i < size; i++)                                          \
            data_dest[i] = conv_b2a(conv_a2b(data_x[i]) op                     \
                conv_a2b(data_y[i]));                                          \
    }                                                                          \
}

/* ======== ARITHMATIC ACCELERATOR HELPERS END ======== */

/* ======== SUM ======== */

/* Integer types. */
_ACCEL_ARITHMATIC(sumi8, int8_t, +,,)
_ACCEL_ARITHMATIC(sumi16, int16_t, +,,)
_ACCEL_ARITHMATIC(sumi32, int32_t, +,,)
_ACCEL_ARITHMATIC(sumi64, int64_t, +,,)

/* Floating point types. */
_ACCEL_ARITHMATIC(sumf8, _fang_float8_t, +, _FANG_Q2S, _FANG_S2Q)
_ACCEL_ARITHMATIC(sumf16, _fang_float16_t, +, _FANG_H2S, _FANG_S2H)
_ACCEL_ARITHMATIC(sumbf16, _fang_bfloat16_t, +, _FANG_BH2S, _FANG_S2BH)
_ACCEL_ARITHMATIC(sumf32, float, +,,)
_ACCEL_ARITHMATIC(sumf64, double, +,,)

/* ======== SUM END ======== */

/* ======== DIFF ======== */

#define _ACCEL_DIFF(postfix, type, conv_a2b, conv_b2a)                         \
FANG_HOT FANG_FLATTEN static void                                              \
    _fang_dense_accel_##postfix(_fang_cpu_accel_arg_t *restrict arg)           \
{                                                                              \
    _ACCEL_ARITH_PROLOGUE(type);                                               \
                                                                               \
    /* Tensor may got swapped to help with broadcasting. */                    \
    int swapped = (swapb_mask >> 0x08) & 0x01;                                 \
                                                                               \
    /* Scalar tensor operation against N-dimensional tensor. */                \
    if(FANG_LIKELY(broadcast == FANG_BCAST_SCALAR)) {                          \
        if(FANG_UNLIKELY(swapped)) {                                           \
            for(int i = 0; i < size; i++)                                      \
                data_dest[i] = conv_b2a(conv_a2b(data_y[0]) -                  \
                    conv_a2b(data_x[i]));                                      \
        } else {                                                               \
            for(int i = 0; i < size; i++)                                      \
                data_dest[i] = conv_b2a(conv_a2b(data_x[i]) -                  \
                    conv_a2b(data_y[0]));                                      \
        }                                                                      \
    }                                                                          \
    /* Row-major vector/matrix operation against N-dimensional tensor. */      \
    else if(FANG_LIKELY(broadcast == FANG_BCAST_ROWVEC ||                      \
        broadcast == FANG_BCAST_MATRIX))                                       \
    {                                                                          \
        if(FANG_UNLIKELY(swapped)) {                                           \
            for(int i = 0; i < size; i += vsiz) {                              \
                for(int j = 0; j < vsiz; j++)                                  \
                    data_dest[i + j] = conv_b2a(conv_a2b(data_y[j]) -          \
                        conv_a2b(data_x[i + j]));                              \
            }                                                                  \
        } else {                                                               \
            for(int i = 0; i < size; i += vsiz) {                              \
                for(int j = 0; j < vsiz; j++)                                  \
                    data_dest[i + j] = conv_b2a(conv_a2b(data_x[i + j]) -      \
                        conv_a2b(data_y[j]));                                  \
            }                                                                  \
        }                                                                      \
    }                                                                          \
    /* Col-major vector operation against N-dimensional tensor. */             \
    else if(FANG_LIKELY(broadcast == FANG_BCAST_COLVEC)) {                     \
        int xvsiz = x->dims[x->ndims - 1];                                     \
        int skip = xvsiz * vsiz;                                               \
                                                                               \
        if(FANG_UNLIKELY(swapped)) {                                           \
            for(int i = 0; i < size; i += skip) {                              \
                for(int j = 0, idx = 0; j < skip; j += xvsiz, idx++) {         \
                    for(int k = 0; k < xvsiz; k++)                             \
                        data_dest[i + j + k] =                                 \
                            conv_b2a(conv_a2b(data_y[idx]) -                   \
                            conv_a2b(data_x[i + j + k]));                      \
                }                                                              \
            }                                                                  \
        } else {                                                               \
            for(int i = 0; i < size; i += skip) {                              \
                for(int j = 0, idx = 0; j < skip; j += xvsiz, idx++) {         \
                    for(int k = 0; k < xvsiz; k++)                             \
                        data_dest[i + j + k] =                                 \
                            conv_b2a(conv_a2b(data_x[i + j + k]) -             \
                            conv_a2b(data_y[idx]));                            \
                }                                                              \
            }                                                                  \
        }                                                                      \
    }                                                                          \
    /* Broadcast dimension unknown. */                                         \
    else if(FANG_UNLIKELY(broadcast == FANG_BCAST_UNKNOWN)) {                  \
        if(FANG_UNLIKELY(swapped)) {                                           \
            for(int i = 0; i < size; i++) {                                    \
                int idx_x, idx_y;                                              \
                /* Get broadcasted flattened index for x and y. */             \
                _fang_get_original_idx(i, &idx_x, &idx_y, dest->strides,       \
                    x->strides, y->strides, x->ndims);                         \
                data_dest[i] = conv_b2a(conv_a2b(data_y[idx_y]) -              \
                    conv_a2b(data_x[idx_x]));                                  \
            }                                                                  \
        } else {                                                               \
            for(int i = 0; i < size; i++) {                                    \
                int idx_x, idx_y;                                              \
                /* Get broadcasted flattened index for x and y. */             \
                _fang_get_original_idx(i, &idx_x, &idx_y, dest->strides,       \
                    x->strides, y->strides, x->ndims);                         \
                data_dest[i] = conv_b2a(conv_a2b(data_x[idx_x]) -              \
                    conv_a2b(data_y[idx_y]));                                  \
            }                                                                  \
        }                                                                      \
    } else {                                                                   \
        for(int i = 0; i < size; i++)                                          \
            data_dest[i] = conv_b2a(conv_a2b(data_x[i]) -                      \
                conv_a2b(data_y[i]));                                          \
    }                                                                          \
}

/* Integer types. */
_ACCEL_DIFF(diffi8, int8_t,,)
_ACCEL_DIFF(diffi16, int16_t,,)
_ACCEL_DIFF(diffi32, int32_t,,)
_ACCEL_DIFF(diffi64, int64_t,,)

/* Floating point types. */
_ACCEL_DIFF(difff8, _fang_float8_t, _FANG_Q2S, _FANG_S2Q)
_ACCEL_DIFF(difff16, _fang_float16_t, _FANG_H2S, _FANG_S2H)
_ACCEL_DIFF(diffbf16, _fang_bfloat16_t, _FANG_BH2S, _FANG_S2BH)
_ACCEL_DIFF(difff32, float,,)
_ACCEL_DIFF(difff64, double,,)

/* ======== DIFF END ======== */

/* ======== MUL ======== */

/* Integer types. */
_ACCEL_ARITHMATIC(muli8, int8_t, *,,)
_ACCEL_ARITHMATIC(muli16, int16_t, *,,)
_ACCEL_ARITHMATIC(muli32, int32_t, *,,)
_ACCEL_ARITHMATIC(muli64, int64_t, *,,)

/* Floating point types. */
_ACCEL_ARITHMATIC(mulf8, _fang_float8_t, *, _FANG_Q2S, _FANG_S2Q)
_ACCEL_ARITHMATIC(mulf16, _fang_float16_t, *, _FANG_H2S, _FANG_S2H)
_ACCEL_ARITHMATIC(mulbf16, _fang_bfloat16_t, *, _FANG_BH2S, _FANG_S2BH)
_ACCEL_ARITHMATIC(mulf32, float, *,,)
_ACCEL_ARITHMATIC(mulf64, double, *,,)

/* ======== MUL END ======== */

/* ======== GEMM ======== */

#define _ACCEL_GEMM_PROLOGUE(type)                                   \
    fang_ten_t *dest = (fang_ten_t *) arg->dest;                     \
    fang_ten_t *x    = (fang_ten_t *) arg->x;                        \
    fang_ten_t *y    = (fang_ten_t *) arg->y;                        \
    type *data_dest  = dest->data.dense;                             \
    type *data_x     = x->data.dense;                                \
    type *data_y     = y->data.dense;                                \
    int size         = (int) dest->strides[0] * dest->dims[0];       \
                                                                     \
    int ld_dest      = (int) dest->dims[dest->ndims - 1];            \
    int ld_x         = (int) x->dims[x->ndims - 1];                  \
    int ld_y         = (int) y->dims[y->ndims - 1];                  \
    int dest_matsiz  = ld_dest * dest->dims[dest->ndims - 2];        \
    int x_matsiz     = ld_x * x->dims[x->ndims - 2];                 \
    int y_matsiz     = ld_y * y->dims[y->ndims - 2];                 \
                                                                     \
    int trn_br_mask  = (int) FANG_G2I(arg->z);                       \
    int broadcast    = trn_br_mask & 0xFF;                           \
    int transp_y     = (trn_br_mask >> 0x08) & 0x01;                 \
    int transp_x     = (trn_br_mask >> 0x09) & 0x01;                 \
                                                                     \
    /* Tensors may get swaped to help with broadcasting. This
     * indicates whether we should swap back the tensor
     * evidently swapping the matrices. */                           \
    bool swap = x->dims[x->ndims - 1 - transp_x] !=                  \
        y->dims[y->ndims - 2 + transp_y];                            \
    /* m, n, k */                                                    \
    int m = dest->dims[dest->ndims - 2],                             \
        n = ld_dest,                                                 \
        k = x->dims[dest->ndims - 1 - (swap ^ transp_x)];            \
    /* alpha and beta. */                                            \
    float alpha = FANG_G2F(arg->alpha);                              \
    float beta = FANG_G2F(arg->beta);                                \
                                                                     \
    int vsiz = 0;                                                    \
    /* Ignore inner two dimensions when broadcasting. */             \
    if(FANG_LIKELY(broadcast == FANG_BCAST_ROWVEC))                  \
        vsiz = y->dims[y->ndims - 3];                                \
    else if(FANG_LIKELY(broadcast == FANG_BCAST_COLVEC))             \
        vsiz = y->dims[y->ndims - 4];                                \
    else if(FANG_LIKELY(broadcast == FANG_BCAST_MATRIX))             \
        vsiz = y->dims[y->ndims - 4] * y->dims[y->ndims - 3];


FANG_HOT FANG_FLATTEN static void
    _fang_dense_accel_gemmf32(_fang_cpu_accel_arg_t *restrict arg)
{
    _ACCEL_GEMM_PROLOGUE(float);

    /* CLARIFICATION: Broadcasting for `fang_ten_gemm()` is done by
     * considering the operand matrix as a single element. Hence, the operand
     * matrix is not considered in dimension related computation. */
    /* Hence, when it is said, say, "Scalar tensor GEMM against N-dimensional
     * tensor", that would mean a single matrix is being broadcast througout
     * numerous amount of flattened matrices (tensors can be thought as series
     * of flattened matrices) and a single matrix can be thought of as a single
     * scalar element if each operand matrix is considered a single element. */

    /* Scalar tensor GEMM against N-dimensional tensor. */
    if(FANG_LIKELY(broadcast == FANG_BCAST_SCALAR)) {
        if(FANG_UNLIKELY(swap)) {
            for(int id = 0, ix = 0; id < size; id += dest_matsiz,
                ix += x_matsiz)
            {
                _fang_sgemm(transp_x, transp_y, m, n, k, beta,
                    data_dest + id, ld_dest, alpha, data_y, ld_y,
                    data_x + ix, ld_x);
            }
        } else {
            for(int id = 0, ix = 0; id < size; id += dest_matsiz,
                ix += x_matsiz)
            {
                _fang_sgemm(transp_x, transp_y, m, n, k, beta,
                    data_dest + id, ld_dest, alpha, data_x + ix, ld_x,
                    data_y, ld_y);
            }
        }
    }
    /* Row-major vector/matrix GEMM against N-dimensional tensor. */
    else if(FANG_LIKELY(broadcast == FANG_BCAST_ROWVEC ||
        broadcast == FANG_BCAST_MATRIX))
    {
        /* All matrix operand sizes are not same. Hence, different striding
           should be enforced. */
        int vsiz_dest = vsiz * dest_matsiz;
        int vsiz_x    = vsiz * x_matsiz;

        if(FANG_UNLIKELY(swap)) {
            for(int id = 0, ix = 0; id < size; id += vsiz_dest, ix += vsiz_x) {
                for(int jd = 0, jx = 0, jy = 0; jd < vsiz_dest;
                    jd += dest_matsiz, jx += x_matsiz, jy += y_matsiz)
                {
                    _fang_sgemm(transp_x, transp_y, m, n, k, beta,
                        data_dest + id + jd, ld_dest, alpha, data_y + jy, ld_y,
                        data_x + ix + jx, ld_x);
                }
            }
        } else {
            for(int id = 0, ix = 0; id < size; id += vsiz_dest, ix += vsiz_x) {
                for(int jd = 0, jx = 0, jy = 0; jd < vsiz_dest;
                    jd += dest_matsiz, jx += x_matsiz, jy += y_matsiz)
                {
                    _fang_sgemm(transp_x, transp_y, m, n, k, beta,
                        data_dest + id + jd, ld_dest, alpha, data_x + ix + jx,
                        ld_x, data_y + jy, ld_y);
                }
            }
        }
    }
    /* Col-major vector GEMM against N-dimensional tensor. */
    else if(FANG_LIKELY(broadcast == FANG_BCAST_COLVEC)) {
        int xvsiz = x->dims[x->ndims - 3];
        int skip = xvsiz * vsiz;

        /* This can be thought of as striding through a matrix (assuming operand
           matrices are a single unit) where a column vector is being added. */
        /* Hassle here is different striding is needed for `dest`, `x` and `y`,
           because matrix operand dimension is not same. */
        int skip_dest = skip * dest_matsiz;
        int skip_x    = skip * x_matsiz;

        int xvsiz_dest = xvsiz * dest_matsiz;
        int xvsiz_x    = xvsiz * x_matsiz;

        if(FANG_UNLIKELY(swap)) {
            for(int id = 0, ix = 0; id < size; id += skip_dest, ix += skip_x) {
                for(int jd = 0, jx = 0, jy = 0; jd < skip_dest;
                    jd += xvsiz_dest, jx += xvsiz_x, jy += y_matsiz)
                {
                    for(int kd = 0, kx = 0; kd < xvsiz_dest; kd += dest_matsiz,
                        kx += x_matsiz)
                    {
                        _fang_sgemm(transp_x, transp_y, m, n, k, beta,
                            data_dest + id + jd + kd, ld_dest, alpha,
                            data_y + jy, ld_y, data_x + ix + jx + kx, ld_x);
                    }
                }
            }
        } else {
            for(int id = 0, ix = 0; id < size; id += skip_dest, ix += skip_x) {
                for(int jd = 0, jx = 0, jy = 0; jd < skip_dest;
                    jd += xvsiz_dest, jx += xvsiz_x, jy += y_matsiz)
                {
                    for(int kd = 0, kx = 0; kd < xvsiz_dest; kd += dest_matsiz,
                        kx += x_matsiz)
                    {
                        _fang_sgemm(transp_x, transp_y, m, n, k, beta,
                            data_dest + id + jd + kd, ld_dest, alpha,
                            data_x + ix + jx + kx, ld_x, data_y + jy, ld_y);
                    }
                }
            }
        }
    }
    /* Broadcast dimension unknown. */
    else if(FANG_UNLIKELY(broadcast == FANG_BCAST_UNKNOWN)) {
        if(FANG_UNLIKELY(swap)) {
            for(int id = 0; id < size; id += dest_matsiz) {
                int idx_x, idx_y;
                /* Get broadcasted flattened index for x and y. */
                _fang_get_original_idx(id, &idx_x, &idx_y, dest->strides,
                    x->strides, y->strides, x->ndims - 2);

                _fang_sgemm(transp_x, transp_y, m, n, k, beta,
                    data_dest + id, ld_dest, alpha, data_y + idx_y, ld_y,
                    data_x + idx_x, ld_x);
            }
        } else {
            for(int id = 0; id < size; id += dest_matsiz) {
                int idx_x, idx_y;
                /* Get broadcasted flattened index for x and y. */
                _fang_get_original_idx(id, &idx_x, &idx_y, dest->strides,
                    x->strides, y->strides, x->ndims - 2);

                _fang_sgemm(transp_x, transp_y, m, n, k, beta,
                    data_dest + id, ld_dest, alpha, data_x + idx_x, ld_x,
                    data_y + idx_y, ld_y);
            }
        }
    } else {
        /* No broadcasting here, hence no need to worry about swapping. */
        for(int id = 0, ix = 0, iy = 0; id < size; id += dest_matsiz,
            ix += x_matsiz, iy += y_matsiz)
        {
            _fang_sgemm(transp_x, transp_y, m, n, k, beta,
                data_dest + id, ld_dest, alpha, data_x + ix, ld_x,
                data_y + iy, ld_y);
        }
    }
}

/* ======== GEMM END ======== */

/* ======== SCALE ======== */

#define _ACCEL_SCALE_PROLOGUE(type, prefix)                                  \
    fang_ten_t *ten = (fang_ten_t *) arg->dest;  /* Tensor */                \
    type factor     = (type) FANG_G2##prefix(arg->x);  /* Factor */          \
    int size        = ten->dims == NULL ? 1 :                                \
        (int) (ten->strides[0] * ten->dims[0]);

#define _ACCEL_SCALE(type, tcast, postfix, prefix, conv_a2b, conv_b2a)       \
FANG_HOT FANG_FLATTEN static void                                            \
    _fang_dense_accel_scale##postfix(_fang_cpu_accel_arg_t *restrict arg)    \
{                                                                            \
    _ACCEL_SCALE_PROLOGUE(tcast, prefix);                                    \
    type *data = (type *) ten->data.dense;                                   \
                                                                             \
    for(int i = 0; i < size; i++)                                            \
        data[i] = conv_b2a(factor * conv_a2b(data[i]));                      \
}

/* Integer types. */
_ACCEL_SCALE(int8_t, int8_t, i8, I,,)
_ACCEL_SCALE(int16_t, int16_t, i16, I,,)
_ACCEL_SCALE(int32_t, int32_t, i32, I,,)
_ACCEL_SCALE(int64_t, int64_t, i64, I,,)

/* Floating point types. */
_ACCEL_SCALE(_fang_float8_t, float, f8, F, _FANG_Q2S, _FANG_S2Q)
_ACCEL_SCALE(_fang_float16_t, float, f16, F, _FANG_H2S, _FANG_S2H)
_ACCEL_SCALE(_fang_bfloat16_t, float, bf16, F, _FANG_BH2S, _FANG_S2BH)
_ACCEL_SCALE(float, float, f32, F,,)
_ACCEL_SCALE(double, double, f64, F,,)

/* ======== SCALE END ======== */

/* ======== FILL ======== */

#define _ACCEL_FILL_PROLOGUE(type, prefix)                                   \
    fang_ten_t *ten = (fang_ten_t *) arg->dest;  /* Tensor */                \
    type value      = (type) FANG_G2##prefix(arg->x);  /* Value */           \
    int size        = ten->dims == NULL ? 1 :                                \
        (int) (ten->strides[0] * ten->dims[0]);

#define _ACCEL_FILL(type, tcast, postfix, prefix, conv_a2b)                  \
FANG_HOT FANG_FLATTEN static void                                            \
    _fang_dense_accel_fill##postfix(_fang_cpu_accel_arg_t *restrict arg)     \
{                                                                            \
    _ACCEL_FILL_PROLOGUE(tcast, prefix);                                     \
    type *data = (type *) ten->data.dense;                                   \
                                                                             \
    for(int i = 0; i < size; i++)                                            \
        data[i] = conv_a2b(value);                                           \
}

/* Integer types. */
_ACCEL_FILL(int8_t, int8_t, i8, I,)
_ACCEL_FILL(int16_t, int16_t, i16, I,)
_ACCEL_FILL(int32_t, int32_t, i32, I,)
_ACCEL_FILL(int64_t, int64_t, i64, I,)

/* Floating point types. */
_ACCEL_FILL(_fang_float8_t, float, f8, F, _FANG_S2Q)
_ACCEL_FILL(_fang_float16_t, float, f16, F, _FANG_S2H)
_ACCEL_FILL(_fang_bfloat16_t, float, bf16, F, _FANG_S2BH)
_ACCEL_FILL(float, float, f32, F,)
_ACCEL_FILL(double, double, f64, F,)

/* ======== FILL END ======== */

/* ================ ACCELERATOR FUNCTIONS END ================ */

