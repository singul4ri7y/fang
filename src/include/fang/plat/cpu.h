#ifndef FANG_PLAT_CPU_H
#define FANG_PLAT_CPU_H

#include <fang/memory.h>

/* Represents a single CPU. */
typedef struct _fang_cpu {
    /* Name of the CPU. */
    char *name;

    /* The threads we can use. */
    /* This maybe double the amount of cores if Intel Hyperthreading 
       is enabled. */
// #ifdef __linux__
//     pthread_t *thread;
// #endif
    int nthread;

    /* Number of active cores. */
    int nact;
} _fang_cpu_t;

/* Fundamental structure representing all the CPUs in a machine. */
typedef struct _fang_platform_cpu {
    _fang_cpu_t *cpu;
    int ncpu;
} _fang_platform_cpu_t;

/* ---------------- PRIVATE ---------------- */

/* Creates a CPU platform specific private structure. */
/* The structure itself is explicitly allocated. */

int  _fang_platform_cpu_create(_fang_platform_cpu_t **restrict cpup, 
        fang_reallocator_t realloc);

/* Release CPU platform. */
void _fang_platform_cpu_release(void *restrict private, 
        fang_reallocator_t realloc);

/* ---------------- PRIVATE END ---------------- */

#endif // FANG_PLAT_CPU_H
