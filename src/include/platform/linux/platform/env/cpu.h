#ifndef FANG_LINUX_CPU_H
#define FANG_LINUX_CPU_H

#include <memory.h>

/* ================ TYPES ================ */

/* Thread worker routine in POSIX Thread (Linux). */
typedef void *(*fang_worker_fn)(void *);

/* ================ TYPES END ================ */


/* ================ DATA STRUCTURES ================ */

/* Represents a single physical CPU. */
typedef struct _fang_cpu {
    /* Number of logical processors. This maybe twice CPU core count if
       Hyperthreading is enabled. */
    int nproc;

    /* Processors start index in system. */
    int sproc;

    /* Number of active processors. */
    int nact;
} _fang_cpu_t;

/* ================ DATA STRUCTURES END ================ */


/* ================ DECLARATIONS ================ */

/* Gets CPU information (how many cores, start index etc.) in Linux. */
int _fang_env_cpu_getinfo(_fang_cpu_t **restrict cpu_buff, int *restrict ncpu,
    fang_reallocator_t realloc);

/* ================ DECLARATIONS END ================ */

#endif  // FANG_LINUX_CPU_H
