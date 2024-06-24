#define _GNU_SOURCE
#include <fang/platform.h>
#include <fang/status.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <limits.h>

#if __linux__
#include <pthread.h>
#include <sched.h>
#endif // __linux__

/* ---------------- TRANSLATION DATA STRUCTURES ---------------- */

/* Represents a single CPU. */
typedef struct _fang_cpu {
    /* Tasks per processors. */
    void *tasks;

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
} _fang_platform_cpu_t;

/* These datatypes are used internally for computation. */
typedef union _fang_internal_data {
    double   *f64d;
    float    *f32d;
    int64_t  *i64d;
    int32_t  *i32d;
    int16_t  *i16d;
    int8_t   *i8d;
    uint64_t *u64d;
    uint32_t *u32d;
    uint16_t *u16d;
    uint8_t  *u8d;
} _fang_internal_data_t;

/* These datatypes are used in exposed APIs for external
   representation. E.g. setting input tensors and getting
   data from output tensors. */
typedef union _fang_external_data {
    fang_float *fdata;
    fang_int   *idata;
    fang_uint  *udata;
} _fang_external_data_t;

/* ---------------- TRANSLATION DATA STRUCTURES END ---------------- */

/* ---------------- FORWARD DECLARATIONS ---------------- */

/* Create tensor data. */
int _fang_platform_cpu_ops_create(fang_ten_ops_arg_t *arg, 
        void **restrict dest, size_t ndtyp);

/* Release tensor data. */
static void _fang_platform_cpu_ops_release(fang_ten_ops_arg_t *arg);

/* ---------------- FORWARD DECLARATIONS END ---------------- */

/* TENSOR OPERATION STRUCTURE. */
static fang_ten_ops_t _cpu_ops = {
    .create = _fang_platform_cpu_ops_create,
    .release = _fang_platform_cpu_ops_release
};

/* Creates a CPU platform specific private structure. */
/* The structure itself is explicitly allocated. */
int _fang_platform_cpu_create(void **restrict private, 
    fang_reallocator_t realloc) 
{
    int res = FANG_GENOK;

    _fang_platform_cpu_t *cpu_plat = FANG_CREATE(realloc, 
        sizeof(_fang_platform_cpu_t));

    if(cpu_plat == NULL) {
        res = -FANG_NOMEM;
        goto out;
    }

#ifdef __linux__
    {
        int _BUFSIZ = 1024;
        char buf[_BUFSIZ];
    
        FILE *file = fopen("/proc/cpuinfo", "r");

        if(file == NULL) {
            res = -FANG_NOINFO;
            goto out;
        }

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

                    if(cpus == NULL) {
                        res = -FANG_NOMEM;
                        goto out;
                    }

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

    /* Create resources for thread management. */
    for(int i = 0; i < cpu_plat->ncpu; i++) {
        _fang_cpu_t *cpu = cpu_plat->cpu + i;

        pthread_t *thread = (pthread_t *) FANG_CREATE(realloc, cpu->nproc 
            * sizeof(pthread_t));
        
        if(thread == NULL) {
            res = FANG_NOMEM;
            goto out;
        }

        cpu->tasks = thread;
    }
#endif // __linux__

    *private = (void *) cpu_plat;

out: 
    return res;
}

/* Release CPU platform. */
void _fang_platform_cpu_release(void *restrict private, 
    fang_reallocator_t realloc) 
{
    _fang_platform_cpu_t *cpu_plat = (_fang_platform_cpu_t *) private;

    for(int i = 0; i < cpu_plat->ncpu; i++) 
        FANG_RELEASE(realloc, cpu_plat->cpu->tasks);

    FANG_RELEASE(realloc, cpu_plat->cpu);
    FANG_RELEASE(realloc, cpu_plat);
}

/* Get the CPU platform tensor operation structure. */
void _fang_platform_cpu_get_ops(fang_ten_ops_t **restrict ops) {
    *ops = &_cpu_ops;
}

/* ---------------- OPERATIONAL FUNCTIONS ---------------- */

/* Create tensor data. */
#define DATACPY(internal, type, external)                     \
    for(size_t i = 0; i < arg->size; i++)                     \
        i_data.internal[i] = (type) e_data.external[i];
int _fang_platform_cpu_ops_create(fang_ten_ops_arg_t *arg, 
    void **restrict dest, size_t ndtyp) 
{
    int res = FANG_GENOK;

    fang_platform_t *platform = (fang_platform_t *) arg->plat;
    *dest = FANG_CREATE(platform->realloc, arg->size * ndtyp);

    if(*dest == NULL) {
        res = FANG_NOMEM;
        goto out;
    }

    memset(*dest, 0, sizeof(arg->size * ndtyp));

    /* The 'data' inside 'arg' is the initializer data in this case. */
    if(arg->data != NULL) {
        _fang_internal_data_t i_data;
        /* Set any of the fields. Doesn't matter which one. */
        i_data.f64d = (double *) *dest;

        _fang_external_data_t e_data;
        e_data.idata = (fang_int *) arg->data;

        switch(arg->typ) {
            case FANG_TEN_DTYPE_FLOAT64: {
                DATACPY(f64d, double, fdata);
            } break;

            case FANG_TEN_DTYPE_FLOAT32: {
                DATACPY(f32d, double, fdata);
            } break;

            case FANG_TEN_DTYPE_INT64: {
                DATACPY(i64d, int64_t, idata);
            } break;

            case FANG_TEN_DTYPE_UINT64: {
                DATACPY(u64d, uint64_t, udata);
            } break;

            case FANG_TEN_DTYPE_INT32: {
                DATACPY(i32d, int32_t, idata);
            } break;

            case FANG_TEN_DTYPE_UINT32: {
                DATACPY(u64d, uint32_t, udata);
            } break;

            case FANG_TEN_DTYPE_INT16: {
                DATACPY(i16d, int16_t, idata);
            } break;

            case FANG_TEN_DTYPE_UINT16: {
                DATACPY(u16d, uint16_t, udata);
            } break;

            case FANG_TEN_DTYPE_INT8: {
                DATACPY(i8d, int8_t, idata);
            } break;

            case FANG_TEN_DTYPE_UINT8: {
                DATACPY(u8d, uint8_t, udata);
            } break;

            default: break;
        }
    }

out: 
    return res;
}
#undef DATACPY

/* Release tensor data. */
void _fang_platform_cpu_ops_release(fang_ten_ops_arg_t *arg) {
    fang_platform_t *platform = (fang_platform_t *) arg->plat;
    FANG_RELEASE(platform->realloc, arg->data);
}

/* Fill tensor data with random values. */

/* ---------------- OPERATIONAL FUNCTIONS END ---------------- */
