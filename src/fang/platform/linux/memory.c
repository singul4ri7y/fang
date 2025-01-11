#include <platform/memory.h>
#include <compiler.h>

/* ================ DEFINITIONS ================ */

/* For fast aligned memory allocation for special purposes. For general purpose
   memory (re | de)allocation), kindly use a reallocator. */
void *_fang_aligned_malloc(size_t size, size_t align) {
    void *ptr = NULL;

    if(FANG_UNLIKELY(posix_memalign(&ptr, align, size)))
        ptr = NULL;

    return ptr;
}

/* ================ DEFINITIONS END ================ */
