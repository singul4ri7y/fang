#include <fang/plat/cpu/work.h>
#include <fang/plat/cpu/task.h>
#include <stdint.h>
#include <stdlib.h>

#ifdef FANG_USE_AVX2
#include <immintrin.h>
#endif // FANG_USE_AVX2

/* ---------------- WORKER FUNCTION (KERNEL) ---------------- */

#ifdef __linux__
#include <pthread.h>

#define WORKER_RAND_PROLOGUE(type)                                   \
    _fang_cpu_task_arg_t *task_arg = (_fang_cpu_task_arg_t *) arg;   \
    type diff = _FANG_BITCAST(type, fang_gen, task_arg->a);          \
    type low = _FANG_BITCAST(type, fang_gen, task_arg->b);           \
    type *data = (type *) task_arg->dest;                            \
    uint32_t seed = (uint32_t) time(0) + task_arg->tid;
#define WORKER_RANDF(bit, type)                                      \
void *_fang_worker_fn_randf##bit(void *arg) {                        \
    WORKER_RAND_PROLOGUE(type)                                       \
    for(int i = 0; i < task_arg->size; i++)                          \
        data[i] = ((type) rand_r(&seed) / RAND_MAX) * diff + low;    \
    pthread_exit(NULL);                                              \
}
#define WORKER_RANDI(postfix, type)                                  \
void *_fang_worker_fn_rand##postfix(void *arg) {                     \
    WORKER_RAND_PROLOGUE(type)                                       \
    for(int i = 0; i < task_arg->size; i++)                          \
        data[i] = ((type) rand_r(&seed)) % (diff + 1) + low;         \
    pthread_exit(NULL);                                              \
}
// TODO: Add noret compiler extension.

#define WORKER_ARITH(atype, asign, type)                             \
void *_fang_worker_fn_##atype(void *arg) {                           \
    _fang_cpu_task_arg_t *task_arg = (_fang_cpu_task_arg_t *) arg;   \
    type *dest = (type *) task_arg->dest;                            \
    type *a = (type *) task_arg->a;                                  \
    type *b = (type *) task_arg->b;                                  \
    for(int i = 0; i < task_arg->size; i++)                          \
        dest[i] = a[i] asign b[i];                                   \
    pthread_exit(NULL);                                              \
}

#endif // __linux__

WORKER_RANDF(64, double)
WORKER_RANDF(32, float)
WORKER_RANDI(i64, int64_t)
WORKER_RANDI(i32, int32_t)
WORKER_RANDI(i16, int16_t)
WORKER_RANDI(i8, int8_t)
WORKER_RANDI(u64, uint64_t)
WORKER_RANDI(u32, uint32_t)
WORKER_RANDI(u16, uint16_t)
WORKER_RANDI(u8, uint8_t)

WORKER_ARITH(sumf64, +, double);
WORKER_ARITH(sumf32, +, float);
WORKER_ARITH(sumi64, +, int64_t);
WORKER_ARITH(sumi32, +, int32_t);
WORKER_ARITH(sumi16, +, int16_t);
WORKER_ARITH(sumi8, +, int8_t);
WORKER_ARITH(sumu64, +, uint64_t);
WORKER_ARITH(sumu32, +, uint32_t);
WORKER_ARITH(sumu16, +, uint16_t);
WORKER_ARITH(sumu8, +, uint8_t);

WORKER_ARITH(difff64, -, double);
WORKER_ARITH(difff32, -, float);
WORKER_ARITH(diffi64, -, int64_t);
WORKER_ARITH(diffi32, -, int32_t);
WORKER_ARITH(diffi16, -, int16_t);
WORKER_ARITH(diffi8, -, int8_t);
WORKER_ARITH(diffu64, -, uint64_t);
WORKER_ARITH(diffu32, -, uint32_t);
WORKER_ARITH(diffu16, -, uint16_t);
WORKER_ARITH(diffu8, -, uint8_t);

WORKER_ARITH(hadamardf64, *, double);
WORKER_ARITH(hadamardf32, *, float);
WORKER_ARITH(hadamardi64, *, int64_t);
WORKER_ARITH(hadamardi32, *, int32_t);
WORKER_ARITH(hadamardi16, *, int16_t);
WORKER_ARITH(hadamardi8, *, int8_t);
WORKER_ARITH(hadamardu64, *, uint64_t);
WORKER_ARITH(hadamardu32, *, uint32_t);
WORKER_ARITH(hadamardu16, *, uint16_t);
WORKER_ARITH(hadamardu8, *, uint8_t);

/* ---------------- WORKER FUNCTION (KERNEL) END ---------------- */
