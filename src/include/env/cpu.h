#ifndef FANG_ENV_CPU_H
#define FANG_ENV_CPU_H

#include <fang/env.h>
#include <memory.h>

/* ================ DECLARATIONS ================ */

/* Creates and initializes a CPU specific Environment. */
int _fang_env_cpu_create(fang_env_private_t **restrict private,
    fang_env_ops_t **restrict ops, fang_reallocator_t realloc);

/* ================ DECLARATIONS END ================ */

#endif  // FANG_ENV_CPU_H
