#include <fang/config.h>
#include <memory.h>
#include <compiler.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

/* ================ PRIVATE MACROS ================ */

#define _FANG_METADATA    (sizeof(size_t) + sizeof(void *) + FANG_MEMALIGN)

/* ================ PRIVATE MACROS END ================ */


/* ================ DEFINITIONS ================ */

/* Fang's default reallocator, used as an alternative if no explicit reallocator
   provided. */
/* Way this reallocator function handles pointers:
 *      |----|--|--|-----------------|
 *      ^    ^  ^  ^
 *      1    2  3  4
 * 1    : original pointer
 * 1 - 4: total offset
 * 2 - 3: space to store current size
 * 3 - 4: space to store original pointer
 * 4    : `FANG_MEMALIGN` bytes aligned memory address
 */
void *_fang_default_reallocator(void *buff, size_t size) {
    /* Using `char *` for simplicity purposes. */
    char *res = NULL;

    /* Get original pointer, offset and old size (needed later on). */
    char *ptr        = buff != NULL ? ((void **) buff)[-1] : NULL;
    size_t old_size  = buff != NULL ? ((size_t *) buff)[-2] : 0;
    ptrdiff_t offset = (char *) buff - ptr;

    /* Invalid use of reallocator function. */
    if(FANG_UNLIKELY(buff == NULL && size == 0))
        goto out;

    /* If 'size' is 0, memory should be freed. */
    if(FANG_LIKELY(size == 0)) {
        free(ptr);
        goto out;
    }

    /* If 'buf' is NULL, memory should be allocated. */
    if(FANG_LIKELY(buff == NULL)) {
        /* `sizeof(void *)`: space to store original pointer. */
        /* `sizeof(size_t)`: space to store current size. */
        ptr = malloc(size + _FANG_METADATA);
        goto calc_addr;
    }

    /* On normal occasions, extend/shrink memory. */
    ptr = realloc(ptr, size + _FANG_METADATA);

calc_addr:
    /* Calculate `FANG_MEMALIGN` byte aligned address, while reserving space to
       store the original memory adderss. */
    res = (void *) (((uintptr_t) ptr + _FANG_METADATA - 1)
          & ~(FANG_MEMALIGN - 1));
    /* If previous offset is not same. */
    if(offset && offset != res - ptr)
        memmove(res, ptr + offset, old_size < size ? old_size : size);

    /* Store the actual pointer in the previous slot. */
    ((void **) res)[-1] = ptr;
    /* NOTE: On most architectures size of `size_t` and `void *` is same. */
    ((size_t *) res)[-2] = size;

out:
    return (void *) res;
}

/* ================ DEFINITIONS END ================ */
