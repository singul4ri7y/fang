#ifndef FANG_PLATFORM_MEMORY_H
#define FANG_PLATFORM_MEMORY_H

#include <compiler.h>
#include <stdlib.h>

/* ================ DECLARATIONS ================ */

/* For fast aligned memory allocation for special purposes. For general purpose
   memory (re | de)allocation), kindly use a reallocator. */
FANG_MALLOC FANG_HOT void *_fang_aligned_malloc(size_t size, size_t align);

/* ================ DECLARATIONS END ================ */

#endif  // FANG_PLATFORM_MEMORY_H
