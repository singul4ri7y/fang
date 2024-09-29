#ifndef FANG_LINUX_CPU_H
#define FANG_LINUX_CPU_H

#include <fang/tensor.h>
#include <env/cpu/cpu.h>
#include <memory.h>
#include <compiler.h>

/* ================ DECLARATIONS ================ */

/* Gets CPU information (how many cores, start index etc.) in Linux. */
int _fang_env_cpu_getinfo(_fang_cpu_t **restrict cpu_buff, int *restrict ncpu,
    fang_reallocator_t realloc);

/* ================ DECLARATIONS END ================ */


/* ================ PLATFORM SPECIFIC OPERATORS ================ */

FANG_HOT int _fang_env_cpu_dense_ops_rand(fang_ten_ops_arg_t *restrict arg,
    fang_env_t *env);

/* ================ PLATFORM SPECIFIC OPERATORS END ================ */

#endif  // FANG_LINUX_CPU_H
