#include <fang/platform.h>
#include <fang/plat/cpu/task.h>
#include <fang/plat/cpu/work.h>
#include <fang/plat/cpu/abst.h>
#include <fang/status.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdbool.h>

/* ---------------- HELPER MACROS ---------------- */

/* Helper macro to reduce redundancy. */
#define FANG_CHECK(data, err)             \
    if(data == NULL) {                    \
        res = -err;                       \
        goto out;                         \
    }

#define FANG_PLATFORM_CPU_ARITH_DECL(atype)                                    \
static int _fang_platform_cpu_ops_##atype(fang_ten_ops_arg_t *restrict arg,    \
        void *restrict a, void *restrict b);

/* ---------------- HELPER MACROS END ---------------- */

/* ---------------- TRANSLATION DATA STRUCTURES ---------------- */

/* Represents a single CPU. */
typedef struct _fang_cpu {
    /* Tasks holding task data per processor. */
    _fang_cpu_task_t *task;

    /* The processors (logical CPUs/cores/threads) we can use. */
    /* This maybe double the amount of actual cores if Intel Hyperthreading 
       is enabled. */
    int nproc;

    /* Processor start index in the system. */
    int sproc;

    /* Number of active processors. */
    int nact;
} _fang_cpu_t;

/* Fundamental structure representing all the CPUs in a machine. */
typedef struct _fang_platform_cpu {
    _fang_cpu_t *cpu;
    int ncpu;

    /* Total active processors in the machine. */
    int ncact;
} _fang_platform_cpu_t;

/* ---------------- TRANSLATION DATA STRUCTURES END ---------------- */

/* ---------------- FORWARD DECLARATIONS ---------------- */

/* Create tensor data. */
static int _fang_platform_cpu_ops_create(fang_ten_ops_arg_t *arg, 
        void **restrict dest, size_t ndtyp);

/* Release tensor data. */
static void _fang_platform_cpu_ops_release(fang_ten_ops_arg_t *arg);

/* Print a tensor. */
static int _fang_platform_cpu_ops_print(fang_ten_ops_arg_t *restrict arg, 
    void *restrict data, int padding);

/* Fill tensor data with random values. */
static int _fang_platform_cpu_ops_rand(fang_ten_ops_arg_t *restrict arg, 
     fang_gen low, fang_gen high); 

/* Sum up two tensor data. */
static int _fang_platform_cpu_ops_sum(fang_ten_ops_arg_t *restrict arg,
        void *restrict a, void *restrict b);

/* Find difference between two tensor data. */
static int _fang_platform_cpu_ops_diff(fang_ten_ops_arg_t *restrict arg,
        void *restrict a, void *restrict b);

/* Element wise multiplicaiton (Hadamard product) of two tensor data. */
static int _fang_platform_cpu_ops_hadamard(fang_ten_ops_arg_t *restrict arg,
        void *restrict a, void *restrict b);

/* ---------------- FORWARD DECLARATIONS END ---------------- */

#ifdef __linux__
typedef void *(*_fang_worker_fn)(void *);
#endif // __linux__

/* TENSOR OPERATION STRUCTURE. */
static fang_ten_ops_t _cpu_ops = {
    .create = _fang_platform_cpu_ops_create,
    .release = _fang_platform_cpu_ops_release,
    .print = _fang_platform_cpu_ops_print,
    .rand = _fang_platform_cpu_ops_rand,
    .sum = _fang_platform_cpu_ops_sum,
    .diff = _fang_platform_cpu_ops_diff,
    .hadamard = _fang_platform_cpu_ops_hadamard
};

/* ---------------- DEFINITIONS ---------------- */

/* Creates a CPU platform specific private structure. */
/* The structure itself is explicitly allocated. */
int _fang_platform_cpu_create(void **restrict private, 
    fang_reallocator_t realloc) 
{
    int res = FANG_GENOK;

    _fang_platform_cpu_t *cpu_plat = FANG_CREATE(realloc, 
        sizeof(_fang_platform_cpu_t));
    FANG_CHECK(cpu_plat, FANG_NOMEM)

#ifdef __linux__
    {
        int _BUFSIZ = 1024;
        char buf[_BUFSIZ];
    
        FILE *file = fopen("/proc/cpuinfo", "r");
        FANG_CHECK(file, -FANG_NOINFO);

        _fang_cpu_t *cpus = NULL;
        int curr_id = -1;
        int ncpu    = 0;

        /* Reconnaissance :). */
        while(fgets(buf, _BUFSIZ, file)) {
            /* If we find "physical id : ", try processing it. If we find an ID 
               greater than the previous one, we got a new CPU. */
            if(strncmp(buf, "physical id", 11) == 0) {
                char *colon = strchr(buf, ':');
                if(colon == NULL) 
                    continue;

                int phy_id = atoi(colon + 2); // Skip ": "
                /* We've got ourselves a new physical CPU. */
                if(phy_id > curr_id) {
                    /* In Linux, physical ids are increamenting. */
                    cpus = realloc(cpus, (phy_id + 1) * sizeof(_fang_cpu_t));
                    FANG_CHECK(cpus, FANG_NOMEM);

                    memset(cpus + phy_id, 0, sizeof(_fang_cpu_t));
                    ncpu++;
                    curr_id = phy_id;
                }
            }
            /* We need total number threads a CPU hold. */
            else if(strncmp(buf, "siblings", 8) == 0) {
                char *colon = strchr(buf, ':');
                if(colon && curr_id != -1 && cpus[curr_id].nproc == 0) {
                    int nproc = atoi(colon + 2);

                    cpus[curr_id].nproc = nproc;
                    cpus[curr_id].nact  = nproc;

                    if(curr_id > 0) {
                        int prev_id = curr_id - 1;

                        cpus[curr_id].sproc = cpus[prev_id].sproc 
                            + cpus[prev_id].nproc;
                    } else cpus[curr_id].sproc = 0; // Very first CPU.
                }
            }
        }

        fclose(file);

        cpu_plat->ncpu = ncpu;
        cpu_plat->cpu  = cpus;
    }
#endif // __linux__

    int ncact = 0;
    /* Create resources for thread management and count total active proc. */
    for(int i = 0; i < cpu_plat->ncpu; i++) {
        _fang_cpu_t *cpu = cpu_plat->cpu + i;

        _fang_cpu_task_t *task = (_fang_cpu_task_t *) FANG_CREATE(realloc, 
            sizeof(_fang_cpu_task_t));
        FANG_CHECK(task, FANG_NOMEM);

#ifdef __linux__
        pthread_t *thread = (pthread_t *) FANG_CREATE(realloc, cpu->nproc 
            * sizeof(pthread_t));
        FANG_CHECK(thread, FANG_NOMEM);
        task->thread = thread;
#endif // __linux__

        _fang_cpu_task_arg_t *arg = (_fang_cpu_task_arg_t *) 
            FANG_CREATE(realloc, cpu->nproc * sizeof(_fang_cpu_task_arg_t));
        FANG_CHECK(arg, FANG_NOMEM);
        task->arg = arg;

        cpu->task = task;

        ncact += cpu->nact;
    }

    cpu_plat->ncact = ncact;

    *private = (void *) cpu_plat;

out: 
    return res;
}

/* Release CPU platform. */
void _fang_platform_cpu_release(void *restrict private, 
    fang_reallocator_t realloc) 
{
    _fang_platform_cpu_t *cpu_plat = (_fang_platform_cpu_t *) private;

    for(int i = 0; i < cpu_plat->ncpu; i++) {
        _fang_cpu_t *cpu = cpu_plat->cpu + i;

        FANG_RELEASE(realloc, cpu->task->thread);
        FANG_RELEASE(realloc, cpu->task->arg);
        FANG_RELEASE(realloc, cpu->task);
    }

    FANG_RELEASE(realloc, cpu_plat->cpu);
    FANG_RELEASE(realloc, cpu_plat);
}

/* Get the CPU platform tensor operation structure. */
void _fang_platform_cpu_get_ops(fang_ten_ops_t **restrict ops) {
    *ops = &_cpu_ops;
}

/* ---------------- DEFINITIONS END ---------------- */

/* ---------------- PRIVATE ---------------- */

static int _fang_platform_cpu_execute_task(_fang_platform_cpu_t *cpu_plat, 
    _fang_worker_fn fn) 
{
    int res = FANG_GENOK;

    for(int i = 0; i < cpu_plat->ncpu; i++) {
        _fang_cpu_t cpu = cpu_plat->cpu[i];

        if(!FANG_OK(res = _fang_platform_cpu_thread_execute(cpu.task->thread, 
            (void *) cpu.task->arg, fn, cpu.nact, cpu.sproc))) 
        {
            goto out;
        }

        _fang_platform_cpu_thread_wait(cpu.task->thread, cpu.nact);
    }

out: 
    return res;
}

/* Recursive solution for tensor printing. */
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
                printf("%11ld", ((int64_t *) data)[stride]);
                break;

            case FANG_TEN_DTYPE_UINT64: 
                printf("%10lu", ((uint64_t *) data)[stride]);
                break;

            case FANG_TEN_DTYPE_INT32: 
                printf("%11d", ((int32_t *) data)[stride]);
                break;

            case FANG_TEN_DTYPE_UINT32: 
                printf("%10u", ((uint32_t *) data)[stride]);
                break;

            case FANG_TEN_DTYPE_INT16: 
                printf("%6hd", ((int16_t *) data)[stride]);
                break;

            case FANG_TEN_DTYPE_UINT16: 
                printf("%5hu", ((uint16_t *) data)[stride]);
                break;

            case FANG_TEN_DTYPE_INT8: 
                printf("%4hhd", ((int8_t *) data)[stride]);
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

    /* Some spacing. */
    if(level == 0) 
        printf("%*s", padding, "");
    else if(indicies[level - 1] != 0) 
        printf("%*s", level + padding, "");

    printf("[");

    for(indicies[level] = 0; indicies[level] < dim; indicies[level]++) {
        _fang_ten_print_rec(typ, data, level + 1, sdims, ndims, 
            indicies, padding, (indicies[level] + 1 == dim));
    }

    /* We are at the final dimension. */
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

/* ---------------- OPERATIONAL FUNCTIONS ---------------- */

/* Create tensor data. */
#define DATACPY(itype, etype)                                      \
for(size_t i = 0; i < arg->size; i++)                              \
    ((itype *) *dest)[i] = (itype) ((etype *) arg->data)[i];
static int _fang_platform_cpu_ops_create(fang_ten_ops_arg_t *arg, 
    void **restrict dest, size_t ndtyp) 
{
    int res = FANG_GENOK;

    fang_platform_t *platform = (fang_platform_t *) arg->plat;
    *dest = FANG_CREATE(platform->realloc, arg->size * ndtyp);

    if(*dest == NULL) {
        res = FANG_NOMEM;
        goto out;
    }

    /* The 'data' inside 'arg' is the initializer data in this case. */
    if(arg->data != NULL) {
        switch(arg->typ) {
            case FANG_TEN_DTYPE_FLOAT64: {
                DATACPY(double, fang_float);
            } break;

            case FANG_TEN_DTYPE_FLOAT32: {
                DATACPY(float, fang_float);
            } break;

            case FANG_TEN_DTYPE_INT64: {
                DATACPY(int64_t, fang_int);
            } break;

            case FANG_TEN_DTYPE_INT32: {
                DATACPY(int32_t, fang_int);
            } break;

            case FANG_TEN_DTYPE_INT16: {
                DATACPY(int16_t, fang_int);
            } break;

            case FANG_TEN_DTYPE_INT8: {
                DATACPY(int8_t, fang_int);
            } break;

            case FANG_TEN_DTYPE_UINT64: {
                DATACPY(uint64_t, fang_uint);
            } break;

            case FANG_TEN_DTYPE_UINT32: {
                DATACPY(uint32_t, fang_uint);
            } break;

            case FANG_TEN_DTYPE_UINT16: {
                DATACPY(uint16_t, fang_uint);
            } break;

            case FANG_TEN_DTYPE_UINT8: {
                DATACPY(uint8_t, fang_uint);
            } break;

            default: break;
        }
    } else memset(*dest, 0, arg->size * ndtyp);

out: 
    return res;
}
#undef DATACPY

/* Release tensor data. */
static void _fang_platform_cpu_ops_release(fang_ten_ops_arg_t *restrict arg) {
    fang_platform_t *platform = (fang_platform_t *) arg->plat;
    FANG_RELEASE(platform->realloc, arg->data);
}

/* Print a tensor. */
static int _fang_platform_cpu_ops_print(fang_ten_ops_arg_t *restrict arg, 
    void *restrict data, int padding) 
{
    int res = FANG_GENOK;

    if(arg->typ == FANG_TEN_DTYPE_INVALID) {
        res = -FANG_INVTYP;
        goto out;
    }

    fang_platform_t *platform = (fang_platform_t *) arg->plat;

    /* Allocate the indicies. */
    uint32_t *indicies = FANG_CREATE(platform->realloc, arg->size * sizeof(uint32_t));
    FANG_CHECK(indicies, FANG_NOMEM);

    _fang_ten_print_rec(arg->typ, data, 0, (uint32_t *) arg->data, arg->size, 
        indicies, padding, false);

    FANG_RELEASE(platform->realloc, indicies);

out: 
    return res;
}

/* Helper macro specific to this function. */
#define CASE(type, dtype, gtype, ndtype, wfn_postfix)      \
case FANG_TEN_DTYPE_##type: {                              \
    dtype hi = (dtype) FANG_G2##gtype(high);               \
    dtype lo = (dtype) FANG_G2##gtype(low);                \
    hi -= lo;                                              \
    a = _FANG_BITCAST(fang_gen, dtype, hi);                \
    b = _FANG_BITCAST(fang_gen, dtype, lo);                \
    runnable = _fang_worker_fn_rand##wfn_postfix;          \
    ndtyp = ndtype;                                        \
} break
/* Fill tensor data with random values. */
static int _fang_platform_cpu_ops_rand(fang_ten_ops_arg_t *restrict arg, 
     fang_gen low, fang_gen high) 
{
    int res = FANG_GENOK;

    fang_platform_t *platform = (fang_platform_t *) arg->plat;
    _fang_platform_cpu_t *cpu_plat = (_fang_platform_cpu_t *) platform->private;

    fang_gen a = NULL; // (high - low)
    fang_gen b = NULL; // low
    _fang_worker_fn runnable;
    int ndtyp = 0;

    switch(arg->typ) {
        CASE(FLOAT64, double, F, 8, f64);
        CASE(FLOAT32, float, F, 4, f32);
        CASE(INT64, int64_t, I, 8, i64);
        CASE(INT32, int32_t, I, 4, i32);
        CASE(INT16, int16_t, I, 2, i16);
        CASE(INT8, int8_t, I, 1, i8);
        CASE(UINT64, uint64_t, U, 8, u64);
        CASE(UINT32, uint32_t, U, 4, u32);
        CASE(UINT16, uint16_t, U, 2, u16);
        CASE(UINT8, uint8_t, U, 1, u8);

        default: {
            res = -FANG_INVTYP;
            goto out;
        }
    }

    /* Amount of computation each processor has to do. */
    int load_per_proc = arg->size / cpu_plat->ncact;
    int load_remain = arg->size % cpu_plat->ncact;

    /* Prepare argument structure. */
    for(int i = 0; i < cpu_plat->ncpu; i++) {
        _fang_cpu_t cpu = cpu_plat->cpu[i];
        for(int j = 0; j < cpu.nact; j++, load_remain--) {
            _fang_cpu_task_arg_t *task_arg = cpu.task->arg + j;

            task_arg->tid = j;
            task_arg->dest = arg->data;
            task_arg->size = load_per_proc + (load_remain > 0);
            task_arg->a = a;
            task_arg->b = b;

            arg->data += (task_arg->size * ndtyp);
        }
    }

    res = _fang_platform_cpu_execute_task(cpu_plat, runnable);

out: 
    return res;        
}
#undef CASE

#define FANG_CPU_PLATFORM_ARITH(atype)                                              \
static int _fang_platform_cpu_ops_##atype(fang_ten_ops_arg_t *restrict arg,         \
        void *restrict a, void *restrict b)                                         \
{                                                                                   \
    int res = FANG_GENOK;                                                           \
    fang_platform_t *platform = (fang_platform_t *) arg->plat;                      \
    _fang_platform_cpu_t *cpu_plat = (_fang_platform_cpu_t *) platform->private;    \
    _fang_worker_fn runnable;                                                       \
    int ndtyp = 0;                                                                  \
    switch(arg->typ) {                                                              \
        CASE(FLOAT64, 8, f64);                                                      \
        CASE(FLOAT32, 4, f32);                                                      \
        CASE(INT64, 8, i64);                                                        \
        CASE(INT32, 4, i32);                                                        \
        CASE(INT16, 2, i16);                                                        \
        CASE(INT8, 1, i8);                                                          \
        CASE(UINT64, 8, u64);                                                       \
        CASE(UINT32, 4, u32);                                                       \
        CASE(UINT16, 2, u16);                                                       \
        CASE(UINT8, 1, u8);                                                         \
        default: {                                                                  \
            res = -FANG_INVTYP;                                                     \
            goto out;                                                               \
        }                                                                           \
    }                                                                               \
    /* Amount of computation each processor has to do. */                           \
    int load_per_proc = arg->size / cpu_plat->ncact;                                \
    int load_remain = arg->size % cpu_plat->ncact;                                  \
    /* Prepare argument structure. */                                               \
    for(int i = 0; i < cpu_plat->ncpu; i++) {                                       \
        _fang_cpu_t cpu = cpu_plat->cpu[i];                                         \
        for(int j = 0; j < cpu.nact; j++, load_remain--) {                          \
            _fang_cpu_task_arg_t *task_arg = cpu.task->arg + j;                     \
            task_arg->tid = j;                                                      \
            task_arg->dest = arg->data;                                             \
            task_arg->size = load_per_proc + (load_remain > 0);                     \
            task_arg->a = (fang_gen) a;                                             \
            task_arg->b = (fang_gen) b;                                             \
            arg->data += (task_arg->size * ndtyp);                                  \
            a += (task_arg->size * ndtyp);                                          \
            b += (task_arg->size * ndtyp);                                          \
        }                                                                           \
    }                                                                               \
    res = _fang_platform_cpu_execute_task(cpu_plat, runnable);                      \
out:                                                                                \
    return res;                                                                     \
}                                                   
                                                    
#define CASE(type, bytes, wfn_type)                \
case FANG_TEN_DTYPE_##type: {                      \
    ndtyp = bytes;                                 \
    runnable = _fang_worker_fn_sum##wfn_type;      \
} break                                             
/* Sum up two tensor data. */
FANG_CPU_PLATFORM_ARITH(sum);
#undef CASE

#define CASE(type, bytes, wfn_type)                \
case FANG_TEN_DTYPE_##type: {                      \
    ndtyp = bytes;                                 \
    runnable = _fang_worker_fn_diff##wfn_type;     \
} break                                             
/* Find difference between two tensor data. */
FANG_CPU_PLATFORM_ARITH(diff);
#undef CASE

#define CASE(type, bytes, wfn_type)                \
case FANG_TEN_DTYPE_##type: {                      \
    ndtyp = bytes;                                 \
    runnable = _fang_worker_fn_hadamard##wfn_type;     \
} break                                             
/* Element wise multiplicaiton (Hadamard product) of two tensor data. */
FANG_CPU_PLATFORM_ARITH(hadamard);
#undef CASE

/* ---------------- OPERATIONAL FUNCTIONS END ---------------- */
