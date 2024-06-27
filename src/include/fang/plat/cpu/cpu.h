#ifndef FANG_PLAT_CPU_H
#define FANG_PLAT_CPU_H

#include <fang/memory.h>

/* ---------------- PRIVATE ---------------- */

/* Creates a CPU platform specific private structure. */
/* The structure itself is heap allocated. */
int  _fang_platform_cpu_create(void **restrict private, 
        fang_reallocator_t realloc);

/* Release CPU platform and related resources. */
void _fang_platform_cpu_release(void *restrict private, 
        fang_reallocator_t realloc);

/* Get the CPU platform tensor operation structure. */
void _fang_platform_cpu_get_ops(fang_ten_ops_t **restrict ops);

/* ---------------- PRIVATE END ---------------- */

#endif // FANG_PLAT_CPU_H
