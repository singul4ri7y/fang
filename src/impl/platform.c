#include <fang/platform.h>
#include <fang/status.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

/* ---------------- PRIVATE ---------------- */

/* All the platforms we are currently using. */
static fang_platform_t _s_platforms[FANG_MAX_PLATFORMS] = {
    [0 ... FANG_MAX_PLATFORMS - 1] = { .type = FANG_PLATFORM_TYPE_INVALID }
};

/* This fn ensures you get the direct reference of a platform structure from 
   the platforms pool. Use with your own risk. */
int _fang_platform_get(fang_platform_t **restrict plat, uint16_t pid) {
    int res = FANG_GENOK;

    if(pid >= FANG_MAX_PLATFORMS) {
        res = -FANG_NOPL;
        goto out;
    }

    /* Such platform exists? */
    if(_s_platforms[pid].type == FANG_PLATFORM_TYPE_INVALID) {
        res = -FANG_NOPL;
        goto out;
    }

    *plat = _s_platforms + pid;

out: 
    return res;
}

/* Default reallocator of Fang, used as an alternative when there is no 
   allocator explicitly used. */
static void *_fang_default_reallocator(void *buff, size_t size) {
    void *res = NULL;

    /* Both invalid 'buff' and 'size' is not acceptable. */
    if(buff == NULL && size == 0) 
        goto out;
    
    /* If 'size' is 0, we are trying to free the memory. */
    if(size == 0) {
        free(buff);
        goto out;
    }

    /* If 'buf' is NULL, we are trying to allocate the memory. */
    if(buff == NULL) {
        res = malloc(size);
        goto out;
    }

    /* On normal occasions, we are trying to extend/shrink memory. */
    res = realloc(buff, size);

out: 
    return res;
}

/* ---------------- PRIVATE END ---------------- */

/* ---------------- DEFINITIONS ---------------- */

int fang_platform_create(fang_platform_type_t type, 
    fang_reallocator_t realloc) 
{
    /* This is our Platform ID. */
    int res = -FANG_NOPL;

    /* If the allocator is NULL, use the default allocator. */
    if(realloc == NULL) 
        realloc = _fang_default_reallocator;

    if(type < 0 || type >= FANG_PLATFORM_TYPE_INVALID) 
        goto out;

    /* Probe and see if there is any room for new platform. */
    for(int i = 0; i < FANG_MAX_PLATFORMS; i++) {
        /* Gotcha. */
        if(_s_platforms[i].type == FANG_PLATFORM_TYPE_INVALID) {
            res = i;
            break;
        }
    }

    /* If we found no slot. */
    if(!FANG_OK(res)) 
        goto out;

    fang_platform_t *plat = _s_platforms + res;

    memset(plat, 0, sizeof(fang_platform_t));

    plat -> type    = type;
    plat -> ntens   = 0;
    plat -> realloc = realloc;

    switch(type) {
        case FANG_PLATFORM_TYPE_CPU: {
            _fang_platform_cpu_t *cpu_plat;

            {
                int code;
                if(!FANG_OK(code = 
                    _fang_platform_cpu_create(&cpu_plat, realloc))) 
                {
                    res = code;
                    goto out;
                }
            }

            plat -> private = (void *) cpu_plat;
            plat -> release = _fang_platform_cpu_release;
        } break;

        default: {
            res = -FANG_NOPL;
            goto out;
        } break;
    }
    
out: 
    return res;
}

int fang_platform_release(uint16_t pid) {
    int res = -FANG_GEN;

    if(pid >= FANG_MAX_PLATFORMS) {
        res = -FANG_NOID;
        goto out;
    }

    fang_platform_t *plat = _s_platforms + pid;

    /* Has the platform already been freed? */
    if(plat -> type == FANG_PLATFORM_TYPE_INVALID) {
        res = -FANG_NOPL;
        goto out;
    }

    /* We cannot release a platform when tensors are allocated to it. */
    if(plat -> ntens != 0) {
        res = -FANG_NTEN;
        goto out;
    }

    plat -> release(plat -> private, plat -> realloc);
    plat -> type = FANG_PLATFORM_TYPE_INVALID;

out: 
    return res;
}

/* ---------------- DEFINITIONS END ---------------- */

