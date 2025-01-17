#ifndef FANG_PLATFORM_CPU_H
#define FANG_PLATFORM_CPU_H

#include <fang/tensor.h>
#include <env/cpu/cpu.h>
#include <memory.h>
#include <compiler.h>

/* ================ DECLARATIONS ================ */

/* Gets CPU information (how many cores, start index etc.) in Linux. */
int _fang_env_cpu_getinfo(int *restrict nproc);

/* ================ DECLARATIONS END ================ */

#endif  // FANG_PLATFORM_CPU_H
