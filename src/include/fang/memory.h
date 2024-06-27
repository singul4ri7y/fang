#ifndef FANG_MEMORY_H
#define FANG_MEMORY_H

#include <stddef.h>

/* Default memory alignement used in Fang (in bytes). */
/* If you intend to use your own reallocator, make sure it's 
   aligned in the following bytes boundary. 64-byte is good 
   for cache lines, specially in con-current environment. */
#define FANG_MEMALIGN    64

/* The reallocator function signature. */
/* If 'buff' is NULL, the function allocates 'size' amount of bytes. */
/* If 'size' is NULL, the function frees 'buff'. */
/* If 'buff' and 'size' both are defined, the function reallocates 'buff'. */
/* If 'buff' and 'size' both are 0 (NULL or invalid), returns null pointer. */ 
typedef void *(*fang_reallocator_t)(void *buff, size_t size);

/* Helper macros to do the niceties when the reallocator is used as pure 
   allocator and deallocator. */
#define FANG_CREATE(realloc, size)     (realloc)(NULL, (size))
#define FANG_RELEASE(realloc, mem)     (realloc)((mem), 0)

#endif // FANG_MEMORY_H
