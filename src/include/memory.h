#ifndef FANG_MEMORY_H
#define FANG_MEMORY_H

#pragma once

#include <memory.h>
#include <compiler.h>
#include <stddef.h>

/* ================ HELPER MACROS ================ */

/* Helper macros to use the reallocator function as pure allocator and
   deallocator. */
#define FANG_CREATE(realloc, type, size)    (realloc)(NULL, (size) * sizeof(type))
#define FANG_RELEASE(realloc, mem)          (realloc)((mem), 0)

/* ================ HELPER MACROS END ================ */


/* ================ TYPES ================ */

/* The reallocator function signature. */
/* If 'buff' is NULL, the function allocates 'size' amount of bytes. */
/* If 'size' is NULL, the function frees 'buff'. */
/* If 'buff' and 'size' both are defined (not NULL), the function reallocates
   'buff'. */
/* If 'buff' and 'size' both are 0 (NULL), it returns NULL pointer. */
typedef void *(*fang_reallocator_t)(void *buff, size_t size);

/* ================ TYPES END ================ */


/* ================ DECLARATIONS ================ */

/* Fang's default reallocator, used as an alternative if no explicit reallocator
   provided. */
FANG_MALLOC FANG_HOT void *_fang_default_reallocator(void *buff, size_t size);

/* ================ DECLARATIONS END ================ */

#endif  // FANG_MEMORY_H
