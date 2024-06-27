#ifndef FANG_PLAT_CPU_ABST_H
#define FANG_PLAT_CPU_ABST_H

#include <fang/plat/cpu/task.h>

/* ---------------- PRIVATE ---------------- */

/* Create threads/tasks and execute the worker functions. */
int _fang_platform_cpu_thread_execute(void *trdata, void *argdata, 
    _fang_worker_fn fn, int nact, int sproc);

/* Wait for all the threads/tasks to exit. */
void _fang_platform_cpu_thread_wait(void *trdata, int nact);

/* ---------------- PRIVATE END ---------------- */

#endif // FANG_PLAT_CPU_ABST_H
