#include <fang/status.h>
#include <env/cpu.h>
#include <platform/env/cpu.h>

/* ================ PRIVATE GLOBALS ================ */

/* Tensor operators for CPU Environment. */
static fang_env_ops_t _cpu_ops = {

};

/* ================ PRIVATE GLOBALS END ================ */


/* ================ PRIVATE DATA STRUCTURES ================ */

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

/* ================ PRIVATE DATA STRUCTURES END ================ */


/* ================ PRIVATE DEFINITIONS ================ */

/* Fully releases a private CPU Environment structures. */
void _fang_env_cpu_release(void *restrict private, fang_reallocator_t realloc) {
    _fang_env_cpu_t *cpu_private = (_fang_env_cpu_t *) private;

    FANG_RELEASE(realloc, cpu_private->cpu);
    FANG_RELEASE(realloc, cpu_private);
}

/* ================ PRIVATE DEFINITIONS END ================ */


/* ================ DEFINITIONS ================ */

/* Creates and initializes a CPU specific Environment. */
int _fang_env_cpu_create(fang_env_private_t **restrict private,
    fang_env_ops_t **restrict ops, fang_reallocator_t realloc)
{
    int res = FANG_OK;

    _fang_env_cpu_t *cpu_private = FANG_CREATE(realloc, _fang_env_cpu_t, 1);
    cpu_private->private.release = _fang_env_cpu_release;

    if(!FANG_ISOK(res = _fang_env_cpu_getinfo(&cpu_private->cpu,
        &cpu_private->ncpu, realloc)))
    {
        goto out;
    }

    /* All physical CPUs are active by default. */
    cpu_private->ncact = cpu_private->ncpu;

    *private = (fang_env_private_t *) cpu_private;
    *ops = &_cpu_ops;

out:
    return res;
}

/* ================ DEFINITIONS END ================ */
