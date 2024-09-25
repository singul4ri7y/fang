#ifndef FANG_LINUX_CPU_H
#define FANG_LINUX_CPU_H

#include <env/cpu/cpu.h>
#include <memory.h>

/* ================ DECLARATIONS ================ */

/* Gets CPU information (how many cores, start index etc.) in Linux. */
int _fang_env_cpu_getinfo(_fang_cpu_t **restrict cpu_buff, int *restrict ncpu,
    fang_reallocator_t realloc);

/* ================ DECLARATIONS END ================ */

#endif  // FANG_LINUX_CPU_H
