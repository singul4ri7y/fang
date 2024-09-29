#define _GNU_SOURCE
#include <fang/status.h>
#include <fang/util/buffer.h>
#include <platform/env/cpu.h>
#include <env/cpu/float.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <pthread.h>
#include <sched.h>

// Taboo lol
#include "dense.c.inc"

/* ================ PRIVATE TYPES ================ */

/* A single kernel (thread worker fn) type for Linux. */
typedef void *(*_fang_task_kernel_t)(void *);

/* ================ PRIVATE TYPES END ================ */


/* ================ PRIVATE GLOBALS ================ */

/* NOTE: Array order conforms to `fang_ten_dtype_t` enum. */
_fang_task_kernel_t _dense_rand[] = {
    _TASKS_DENSE(rand)
};

/* To check difference in tensor randomizer. */
/* NOTE: Array conforms to `fang_ten_dtype_t` enum order. */
fang_uint _of_ma[] = { INT8_MAX, INT16_MAX, INT32_MAX, INT64_MAX,
    UINT8_MAX, UINT16_MAX, UINT32_MAX, UINT64_MAX };  // Max
fang_uint _of_mi[] = { INT8_MIN, INT16_MIN, INT32_MIN, INT64_MIN };  // Min

/* ================ PRIVATE GLOBALS END ================ */


/* ================ PRIVATE DEFINITIONS ================ */

/* Executes worker/kernel functions per processor. */
FANG_HOT static int
    _fang_env_cpu_run_kernel(void *pool, _fang_cpu_task_arg_t *arg,
    _fang_task_kernel_t kernel, int nact, int sproc)
{
    int res = FANG_OK;

    pthread_t *thread = (pthread_t *) pool;

    cpu_set_t cpuset;
    for(int i = 0; i < nact; i++) {
        if(FANG_UNLIKELY(pthread_create(thread + i, NULL, kernel,
            (void *) (arg + i)) != 0))
        {
            res = -FANG_NOTASK;
            goto out;
        }

        CPU_ZERO(&cpuset);
        CPU_SET(sproc + i, &cpuset);
        /* Pin to specific processor (core). */
        if(FANG_UNLIKELY(pthread_setaffinity_np(thread[i], sizeof(cpuset),
            &cpuset) != 0))
        {
            res = -FANG_NOPIN;
            goto out;
        }
    }

out:
    return res;
}

/* Waits for the kernel/worker functions to finish in every processor. */
void _fang_env_cpu_kernel_wait(void *pool, int nact) {
    pthread_t *thread = (pthread_t *) pool;
    for(int j = 0; j < nact; j++)
        pthread_join(thread[j], NULL);
}

/* Prepares and executes tasks on processors. */
FANG_HOT static int _fang_env_cpu_exec_task(_fang_env_cpu_t *restrict cpu_env,
    _fang_cpu_task_arg_t *restrict arg, _fang_task_kernel_t kernel)
{
    int res = FANG_OK;

    fang_ten_t *ten = (fang_ten_t *) arg->dest;
    int size = ten->sdims[0] * ten->sdims[ten->ndims - 1];

    /* Computation load per processor. */
    int load_per_proc = size / cpu_env->ncact;
    int load_remain = size % cpu_env->ncact;

    /* Accumulate stride. */
    size_t stride = 0;
    for(int i = 0; i < cpu_env->ncpu; i++) {
        _fang_cpu_t *cpu = cpu_env->cpu + i;
        for(int j = 0; j < cpu->nact; j++) {
            _fang_cpu_task_arg_t *task_arg = cpu->task->arg + j;
            *task_arg = *arg;

            /* Set task ID. */
            task_arg->tid = j;
            /* Load. */
            task_arg->load = load_per_proc;
            /* Stride. */
            task_arg->stride = stride;

            if(FANG_LIKELY(load_remain > 0)) {
                task_arg->load++;
                load_remain--;
            }

            stride += task_arg->load;
        }

        if(FANG_UNLIKELY(!FANG_ISOK(res = _fang_env_cpu_run_kernel(
            cpu->task->pool, cpu->task->arg, kernel, cpu->nact, cpu->sproc))))
        {
            goto out;
        }
    }

    /* Wait for tasks to complete. */
    for(int i = 0; i < cpu_env->ncpu; i++) {
        _fang_env_cpu_kernel_wait(cpu_env->cpu[i].task->pool,
            cpu_env->cpu[i].nact);
    }

out:
    return res;
}

/* ================ PRIVATE DEFINITIONS END ================ */


/* ================ DEFINITIONS ================ */

/* Private macro for `_fang_env_cpu_getinfo`. */
#define _FANG_CHECK(data)    \
if(data == NULL) { res = -FANG_NOMEM; goto release; }
/* Gets CPU information (how many cores, start index etc.) in Linux. */
int _fang_env_cpu_getinfo(_fang_cpu_t **restrict cpu_buff, int *restrict ncpu,
    fang_reallocator_t realloc)
{
    int res = FANG_OK;

    int _BUFSIZ = 1024;
    char cbuff[_BUFSIZ];

    FILE *info = fopen("/proc/cpuinfo", "r");
    if(info == NULL) {
        res = -FANG_NOINFO;
        goto out;
    }

    fang_buffer_t buff;
    FANG_BUFFER_CREATE(&buff, realloc, _fang_cpu_t);

    int curr_id = -1;

    /* Reconnaissance :). */
    while(fgets(cbuff, _BUFSIZ, info)) {
        /* Potential physical CPU. */
        if(strncmp(cbuff, "physical id", 11) == 0) {
            /* Is it a proper physical ID field? */
            char *colon = strchr(cbuff, ':');
            if(colon == NULL)
                continue;

            /* In Linux, physical CPU IDs are increamenting. */
            int phy_id = atoi(colon + 2);  // Skip ": "
            /* New physical CPU. */
            if(phy_id > curr_id) {
                _fang_cpu_t cpu = { 0 };
                if(!FANG_ISOK(res = fang_buffer_add(&buff, &cpu)))
                    goto release;

                curr_id = phy_id;
            }
        }
        /* Processors count the physical CPU holds. */
        /* It's a good thing "siblings: " field comes after "physical id: "
           field. */
        else if(strncmp(cbuff, "siblings", 8) == 0) {
            /* Proper "siblings: " field? */
            char *colon = strchr(cbuff, ':');
            if(colon && curr_id != -1) {
                _fang_cpu_t *cpu = FANG_BUFFER_GET(&buff, _fang_cpu_t, curr_id);
                if(cpu ->nproc == 0) {
                    int nproc = atoi(colon + 2);  // Skip ": "

                    cpu->nproc = nproc;
                    cpu->nact  = nproc;  // All processors are active by default

                    if(curr_id > 0) {
                        _fang_cpu_t *prev_cpu =
                            FANG_BUFFER_GET(&buff, _fang_cpu_t, curr_id - 1);

                        cpu->sproc = prev_cpu->sproc
                            + prev_cpu->nproc;
                    } else cpu->sproc = 0;  // Very first CPU.
                }
            }
        }
    }

    /* No element would be pushed anymore. */
    fang_buffer_shrink_to_fit(&buff);

    size_t size;
    _fang_cpu_t *cpu = fang_buffer_retrieve(&buff, &size);

    /* Initialize task resources. */
    for(size_t i = 0; i < size; i++) {
        _fang_cpu_task_t *task = FANG_CREATE(realloc, _fang_cpu_task_t, 1);
        _FANG_CHECK(task);

        /* Create task/thread units. */
        task->pool = (void *) FANG_CREATE(realloc, pthread_t, cpu[i].nproc);
        _FANG_CHECK(task->pool);

        /* Create task arguments. */
        task->arg = FANG_CREATE(realloc, _fang_cpu_task_arg_t, cpu[i].nproc);
        _FANG_CHECK(task->arg);

        cpu[i].task = task;
    }

    *cpu_buff = cpu;
    *ncpu = (int) size;

release:
    fclose(info);
out:
    return res;
}

/* ================ DEFINITIONS END ================ */


/* ================ PLATFORM SPECIFIC OPERATORS ================ */

/* Fills a tensor with random numbers. */
int _fang_env_cpu_dense_ops_rand(fang_ten_ops_arg_t *restrict arg,
    fang_env_t *env)
{
    int res = FANG_OK;

    fang_ten_t *ten = (fang_ten_t *) arg->dest;
    _fang_env_cpu_t *cpu_env = (_fang_env_cpu_t *) env->private;

    fang_gen diff, low = arg->x;

    if(FANG_LIKELY(ten->dtyp >= FANG_TEN_DTYPE_INT8 &&
        ten->dtyp <= FANG_TEN_DTYPE_INT64))
    {
        /* high - low */
        /* Add 1 to make high range inclusive. */
        int64_t intm_diff = FANG_G2I(arg->y) - FANG_G2I(arg->x) + 1;

        /* Check for overflow. */
        if(FANG_UNLIKELY(intm_diff < (int64_t) _of_mi[(int) ten->dtyp] ||
            intm_diff > (int64_t) _of_ma[(int) ten->dtyp]))
        {
            res = -FANG_RANDOF;
            goto out;
        }

        diff = FANG_I2G(intm_diff);
    }
    else if(FANG_LIKELY(ten->dtyp >= FANG_TEN_DTYPE_UINT8 &&
        ten->dtyp <= FANG_TEN_DTYPE_UINT64))
    {
        /* high - low */
        uint64_t intm_diff = FANG_G2U(arg->y) - FANG_G2U(arg->x) + 1;

        /* Check for overflow. */
        if(FANG_UNLIKELY(intm_diff >
            (uint64_t) _of_ma[(int) ten->dtyp]))
        {
            res = -FANG_RANDOF;
            goto out;
        }

        diff = FANG_U2G(intm_diff);
    } else diff = FANG_F2G(FANG_G2F(arg->y) - FANG_G2F(arg->x));

    _fang_cpu_task_arg_t task_arg = {
        .dest = (fang_gen) ten,
        .x = diff,
        .y = low,
        .z = arg->z  // seed
    };
    res = _fang_env_cpu_exec_task(cpu_env, &task_arg,
        _dense_rand[(int) ten->dtyp]);

out:
    return res;
}

/* ================ PLATFORM SPECIFIC OPERATORS END ================ */

