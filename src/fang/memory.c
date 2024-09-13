#include <fang/config.h>
#include <memory.h>
#include <compiler.h>
#include <stdlib.h>
#include <stdint.h>

/* Fang's default reallocator, used as an alternative if no explicit reallocator
   provided. */
void *_fang_default_reallocator(void *buff, size_t size) {
    void *res = NULL;

    /* Get the actual pointer. */
    buff = buff != NULL ? ((void **) buff)[-1] : NULL;

    /* Invalid use of reallocator function. */
    if(FANG_UNLIKELY(buff == NULL && size == 0))
        goto out;

    /* If 'size' is 0, memory should be freed. */
    if(FANG_LIKELY(size == 0)) {
        free(buff);
        goto out;
    }

    /* Original memory location. */
    void *ptr = NULL;

    /* If 'buf' is NULL, memory should be allocated. */
    if(FANG_LIKELY(buff == NULL)) {
        ptr = malloc(size + FANG_MEMALIGN + sizeof(void *));
        goto calc_addr;
    }

    /* On normal occasions, extend/shrink memory. */
    ptr = realloc(buff, size + FANG_MEMALIGN + sizeof(void *));

calc_addr:
    /* Calculate `FANG_MEMALIGN` byte aligned address, while reserving space to
     * store the original memory adderss. */
    res = (void *) (((uintptr_t) ptr + sizeof(void *) + FANG_MEMALIGN - 1)
        & ~(FANG_MEMALIGN - 1));

    /* Store the actual pointer in the previous slot. */
    ((void **) res)[-1] = ptr;

out:
    return res;
}
