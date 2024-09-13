#include <fang/status.h>
#include <env/cpu.h>
#include <string.h>

/* ================ PRIVATE ================ */

/* Store all the environment created. */
static fang_env_t _s_envs[FANG_MAX_ENV];

/* ================ PRIVATE END ================ */


/* ================ DEFINITIONS ================ */

/* Creates an Environment and returns the ID. */
int fang_env_create(fang_env_type_t type, fang_reallocator_t realloc) {
    /* This is our Environment ID. */
    int res = -FANG_NOENV;

    if(type < FANG_ENV_TYPE_CPU || type > FANG_ENV_TYPE_GPU)
        goto out;

    /* Use default reallocator. */
    if(realloc == NULL)
        realloc = _fang_default_reallocator;

    /* Probe and see if there is any room for new Environment. */
    for(int i = 0; i < FANG_MAX_ENV; i++) {
        if(_s_envs[i].type == FANG_ENV_TYPE_INVALID) {
            /* Gotcha. */
            res = i;
            break;
        }
    }

    /* No slot was found. */
    if(!FANG_ISOK(res))
        goto out;

    fang_env_t *env = _s_envs + res;
    memset(env, 0, sizeof(fang_env_t));

    env->type = type;
    env->realloc = realloc;

    switch(type) {
        case FANG_ENV_TYPE_CPU: {
            int code;
            if(!FANG_ISOK(code =
                _fang_env_cpu_create(&env->private, &env->ops, realloc)))
            {
                res = code;
                goto out;
            }
        } break;

        case FANG_ENV_TYPE_GPU: {
        } break;

        default: res = -FANG_INVENVTYP;
    }

out:
    return res;
}

/* Releases an Environment if not released. */
int fang_env_release(int env_id) {
    int res = FANG_OK;

    if(env_id >= FANG_MAX_ENV) {
        res = -FANG_INVID;
        goto out;
    }

    fang_env_t *env = _s_envs + env_id;
    /* Nothing to be done if the Environment is already freed. */
    if(env->type == FANG_ENV_TYPE_INVALID)
        goto out;

    /* Environments cannot be released if tensors using it. */
    if(env->ntens != 0) {
        res = -FANG_NTENS;
        goto out;
    }

    env->private->release(env->private, env->realloc);
    env->type = FANG_ENV_TYPE_INVALID;  // Marking the slot free

out:
    return res;
}

/* ================ DEFINITIONS END ================ */

