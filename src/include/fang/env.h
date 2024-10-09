#ifndef FANG_ENV_H
#define FANG_ENV_H

#include <fang/config.h>
#include <fang/tensor.h>
#include <memory.h>
#include <compiler.h>

/* ================ DATA STRUCTURES ================ */

/* Type of Environment in use. */
typedef enum fang_env_type {
    FANG_ENV_TYPE_INVALID,

    FANG_ENV_TYPE_CPU,
    FANG_ENV_TYPE_GPU  // TODO: Implement
} fang_env_type_t;

/* Interface structure to be inherited by Environment private data. */
typedef struct fang_env_private {
    void (*release)(void *restrict private, fang_reallocator_t realloc);
} fang_env_private_t;

/* Hold tensor operators. */
typedef struct fang_env_ops_t {
    fang_ten_ops_t *dense;   // For dense tensors
    fang_ten_ops_t *sparse;  // For sparse tensors
} fang_env_ops_t;

/* Structure of a single Environment. */
typedef struct fang_env {
    /* Type of Environment. */
    fang_env_type_t type;

    /* Number of tensors in this Environment. */
    int ntens;

    /* Reallocator function for CPU specific (de)allocations. */
    fang_reallocator_t realloc;

    /* Environment specific private data. */
    fang_env_private_t *private;

    /* Tensor operators for this Environment. */
    fang_env_ops_t *ops;
} fang_env_t;

/* ================ DATA STRUCTURES END ================ */


/* ================ DECLARATIONS ================ */

/* Creates an Environment and returns the ID. */
FANG_API int fang_env_create(fang_env_type_t type, fang_reallocator_t realloc);

/* Controls number of active processors. */
/* NOTE: Setting `nact` to 0 would active all the processors (cores). */
FANG_API int fang_env_cpu_actproc(int eid, int nact);

/* Releases an Environment if not released. */
FANG_API int fang_env_release(int eid);

/* ================ DECLARATIONS END ================ */


/* ================ PRIVATE DECLARATIONS ================ */

/* Retrieve the Environment structure referenced by the ID from Environment
   pool. REFRAIN FROM USING THIS FUNCTION UNLESS ABSOLUTELY ESSENTIAL. */
FANG_API FANG_HOT int _fang_env_retrieve(fang_env_t **restrict env, int eid);

/* ================ PRIVATE DECLARATIONS END ================ */

#endif  // FANG_ENV_H
