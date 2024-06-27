#define _GNU_SOURCE
#include <fang/plat/cpu/abst.h>
#include <fang/status.h>

#if __linux__
#include <pthread.h>
#include <sched.h>
#endif // __linux__

/* Create threads/tasks and execute the worker functions. */
int _fang_platform_cpu_thread_execute(void *trdata, void *argdata, 
    _fang_worker_fn fn, int nact, int sproc) 
{
    int res = FANG_GENOK;

#ifdef __linux__
    pthread_t *thread = (pthread_t *) trdata;
    _fang_cpu_task_arg_t *arg = (_fang_cpu_task_arg_t *) argdata;

    cpu_set_t cpuset;
    for(int i = 0; i < nact; i++) {
        if(pthread_create(thread + i, NULL,
            fn, (void *) (arg + i)) != 0) 
        {
            res = -FANG_NOTHRD;
            goto out;
        }

        CPU_ZERO(&cpuset);
        CPU_SET(sproc + i, &cpuset);

        /* Pin to specific processor (core). */
        if(pthread_setaffinity_np(thread[i], sizeof(cpuset), &cpuset) != 0) {
            res = -FANG_NOPIN;
            goto out;
        }
    }
#endif // __linux__

out: 
    return res;
}

/* Wait for all the threads/tasks to exit. */
void _fang_platform_cpu_thread_wait(void *trdata, int nact) {
#ifdef __linux__
        pthread_t *thread = (pthread_t *) trdata;
        for(int j = 0; j < nact; j++) 
            pthread_join(thread[j], NULL);
#endif // __linux__
}
