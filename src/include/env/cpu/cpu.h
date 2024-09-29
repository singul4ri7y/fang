#ifndef FANG_ENV_CPU_H
#define FANG_ENV_CPU_H

#include <fang/env.h>
#include <env/cpu/task.h>
#include <memory.h>

/* ================ DATA STRUCTURES ================ */

/* Represents a single physical CPU. */
typedef struct _fang_cpu {
    /* Task data per processor. */
    _fang_cpu_task_t *task;

    /* Number of logical processors. This maybe twice CPU core count if
       Hyperthreading is enabled. */
    int nproc;

    /* Processors start index in system. */
    int sproc;

    /* Number of active processors. */
    int nact;
} _fang_cpu_t;

/* Holds CPU exclusive data. */
typedef struct _fang_env_cpu {
    /* Private structure inheritance. */
    fang_env_private_t private;

    /* Physical CPUs. */
    _fang_cpu_t *cpu;
    int ncpu;

    /* Number of active physical CPUs. */
    int ncact;
} _fang_env_cpu_t;

/* ================ DATA STRUCTURES END ================ */


/* ================ DECLARATIONS ================ */

/* Creates and initializes a CPU specific Environment. */
int _fang_env_cpu_create(fang_env_private_t **restrict private,
    fang_env_ops_t **restrict ops, fang_reallocator_t realloc);

/* Changes processor count of a physical CPU. */
int _fang_env_cpu_actproc(fang_env_private_t *private, int pcpu, int nact);

/* ================ DECLARATIONS END ================ */

#endif  // FANG_ENV_CPU_H
