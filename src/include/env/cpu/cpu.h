#ifndef FANG_ENV_CPU_H
#define FANG_ENV_CPU_H

#include <fang/env.h>
#include <memory.h>

/* ================ DATA STRUCTURES ================ */

/* Holds CPU exclusive data. */
typedef struct _fang_env_cpu {
    /* Private structure inheritance. */
    fang_env_private_t private;

    /* Number of processors in machine. Here, "processor" stands for number of
       logical cores in a machine. This maybe twice if Hyperthreading is
       enabled. */
    int nproc;

    /* Total active processors. */
    int nact;
} _fang_env_cpu_t;

/* ================ DATA STRUCTURES END ================ */


/* ================ DECLARATIONS ================ */

/* Creates and initializes a CPU specific Environment. */
int _fang_env_cpu_create(fang_env_private_t **restrict private,
    fang_env_ops_t **restrict ops, fang_reallocator_t realloc);

/* Changes active processor count. */
int _fang_env_cpu_actproc(fang_env_private_t *private, int nact);

/* ================ DECLARATIONS END ================ */

#endif  // FANG_ENV_CPU_H
