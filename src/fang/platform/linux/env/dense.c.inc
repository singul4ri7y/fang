/* ================ MACROS ================ */

/* Makes life easier. */
/* NOTE: Order conforms to `fang_ten_dtype_t` enum. */
#define _TASKS_DENSE(name)          \
_fang_dense_kernel_##name##i8,      \
_fang_dense_kernel_##name##i16,     \
_fang_dense_kernel_##name##i32,     \
_fang_dense_kernel_##name##i64,     \
_fang_dense_kernel_##name##u8,      \
_fang_dense_kernel_##name##u16,     \
_fang_dense_kernel_##name##u32,     \
_fang_dense_kernel_##name##u64,     \
_fang_dense_kernel_##name##f8,      \
_fang_dense_kernel_##name##f16,     \
_fang_dense_kernel_##name##bf16,    \
_fang_dense_kernel_##name##f32,     \
_fang_dense_kernel_##name##f64

/* ================ MACROS END ================ */


/* ================ KERNEL ================ */

/* ======== RAND ======== */

/* Common. */
#define _KERNEL_RAND_PROLOGUE(type, prefix)                           \
    _fang_cpu_task_arg_t *task_arg = (_fang_cpu_task_arg_t *) arg;    \
    fang_ten_t *ten = (fang_ten_t *) task_arg->dest;  /* Tensor */    \
    type diff = (type) FANG_G2##prefix(task_arg->x);  /* Diff */      \
    type low = (type) FANG_G2##prefix(task_arg->y);  /* Low */        \
    uint32_t seed = (uint32_t) FANG_G2U(task_arg->z) + task_arg->tid;

/* Common for integer types. */
#define _KERNEL_RANDI(prefix, postfix, type, annot)            \
FANG_HOT FANG_FLATTEN void *                                   \
_fang_dense_kernel_rand##postfix(void *arg) {                  \
    _KERNEL_RAND_PROLOGUE(type, prefix);                       \
    type *data = (type *) ten->data.dense;                     \
    for(int i = 0; i < task_arg->load; i++) {                  \
        data[task_arg->stride + i] = (rand_r(&seed) &          \
            annot##_MAX) % diff + low;                         \
    }                                                          \
    pthread_exit(NULL);                                        \
}

/* Common for float types. */
#define _KERNEL_RANDF(bits, type)                               \
FANG_HOT FANG_FLATTEN void *                                    \
_fang_dense_kernel_randf##bits(void *arg) {                     \
    _KERNEL_RAND_PROLOGUE(type, F);                             \
    type *data = (type *) ten->data.dense;                      \
    for(int i = 0; i < task_arg->load; i++) {                   \
        data[task_arg->stride + i] = ((type) rand_r(&seed) /    \
            RAND_MAX) * diff + low;                             \
    }                                                           \
    pthread_exit(NULL);                                         \
}

/* Integer type. */
_KERNEL_RANDI(I, i8, int8_t, INT8)
_KERNEL_RANDI(I, i16, int16_t, INT16)
_KERNEL_RANDI(I, i32, int32_t, INT32)
_KERNEL_RANDI(I, i64, int64_t, INT64)
_KERNEL_RANDI(U, u8, uint8_t, UINT8)
_KERNEL_RANDI(U, u16, uint16_t, UINT16)
_KERNEL_RANDI(U, u32, uint32_t, UINT32)
_KERNEL_RANDI(U, u64, uint64_t, UINT64)

FANG_HOT FANG_FLATTEN void *_fang_dense_kernel_randf8(void *arg) {
    _KERNEL_RAND_PROLOGUE(float, F);
    _fang_float8_t *data = (_fang_float8_t *) ten->data.dense;

    for(int i = 0; i < task_arg->load; i++) {
        data[task_arg->stride + i] = _FANG_S2Q(((float) rand_r(&seed) /
            RAND_MAX) * diff + low);
    }

    pthread_exit(NULL);
}

FANG_HOT FANG_FLATTEN void *_fang_dense_kernel_randf16(void *arg) {
    _KERNEL_RAND_PROLOGUE(float, F);
    _fang_float16_t *data = (_fang_float16_t *) ten->data.dense;

    for(int i = 0; i < task_arg->load; i++) {
        data[task_arg->stride + i] = _FANG_S2H(((float) rand_r(&seed) /
            RAND_MAX) * diff + low);
    }

    pthread_exit(NULL);
}

FANG_HOT FANG_FLATTEN void *_fang_dense_kernel_randbf16(void *arg) {
    _KERNEL_RAND_PROLOGUE(float, F);
    _fang_bfloat16_t *data = (_fang_bfloat16_t *) ten->data.dense;

    for(int i = 0; i < task_arg->load; i++) {
        data[task_arg->stride + i] = _FANG_S2BH(((float) rand_r(&seed) /
            RAND_MAX) * diff + low);
    }

    pthread_exit(NULL);
}

_KERNEL_RANDF(32, float)
_KERNEL_RANDF(64, double)

/* ======== RAND END ======== */

/* ================ KERNEL END ================ */
