#ifndef FANG_PLAT_CPU_TASK_H
#define FANG_PLAT_CPU_TASK_H

#include <fang/type.h>

/* Arguments passed to each worker thread. */
typedef struct _fang_cpu_task_arg {
    /* Parameters. Can be any combinations. */
    void *dest;
    fang_gen a;
    fang_gen b;
    
    /* How much load we have? */
    int size;

    /* Our task id. */
    int tid;
} _fang_cpu_task_arg_t;

/* Tasks all the processors will handle per physical CPU. */
typedef struct _fang_cpu_task {
    void *thread;
    _fang_cpu_task_arg_t *arg;
} _fang_cpu_task_t;

/* Represents a working unit function for parallel workload (kernel). */
#ifdef __linux__
typedef void *(*_fang_worker_fn)(void *);
#endif // __linux__

#endif // FANG_PLAT_CPU_TASK_H
