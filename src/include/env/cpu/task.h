#ifndef FANG_CPU_TASK_H
#define FANG_CPU_TASK_H

#include <stddef.h>

/* ================ DATA STRUCTURES ================ */

/* Arguments passed to each worker task. */
typedef struct _fang_cpu_task_arg {
    /* Task id. */
    int tid;

    /* How much workload a task has? */
    int load;

    /* Start on which point of data? */
    size_t stride;

    /* Parameters. */
    fang_gen dest;  // Probably destination tensor
    /* Can be tensor or general purpose data. */
    fang_gen x;
    fang_gen y;
    fang_gen z;
} _fang_cpu_task_arg_t;

/* Tasks all the processors will handle per processor. */
typedef struct _fang_cpu_task {
    /* Thread/task pool. */
    void *pool;

    /* Argument per task. */
    _fang_cpu_task_arg_t *arg;
} _fang_cpu_task_t;

/* ================ DATA STRUCTURES END ================ */

#endif  // FANG_CPU_TASK_H
