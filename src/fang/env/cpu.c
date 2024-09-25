#include <fang/status.h>
#include <fang/tensor.h>
#include <env/cpu/float.h>
#include <platform/env/cpu.h>
#include <string.h>

/* ================ FORWARD DECLARATIONS ================ */

/* Creates and initializes tensor data. */
FANG_HOT FANG_FLATTEN int
    _fang_env_cpu_ops_create(fang_ten_ops_arg_t *restrict arg);

/* ================ FORWARD DECLARATIONS END ================ */


/* ================ PRIVATE GLOBALS ================ */

static fang_ten_ops_t _dense = {
    .create = _fang_env_cpu_ops_create
};

/* Tensor operators for CPU Environment. */
static fang_env_ops_t _cpu_ops = {
    .dense  = &_dense,
    .sparse = NULL
};

/* Single element data size of each tensor data type. */
/* NOTE: This array conforms to `fang_ten_dtype_t` enum, any changes made to
   that enum should reflect here. */
static const int dsiz[] = { 1, 2, 4, 8 };
static const int ndsiz  = 4;

/* ================ PRIVATE GLOBALS END ================ */


/* ================ PRIVATE DEFINITIONS ================ */

/* Fully releases a private CPU Environment structures. */
void _fang_env_cpu_release(void *restrict private, fang_reallocator_t realloc) {
    _fang_env_cpu_t *cpu_private = (_fang_env_cpu_t *) private;

    FANG_RELEASE(realloc, cpu_private->cpu);
    FANG_RELEASE(realloc, cpu_private);
}

/* ================ PRIVATE DEFINITIONS END ================ */


/* ================ DEFINITIONS ================ */

/* Creates and initializes a CPU specific Environment. */
int _fang_env_cpu_create(fang_env_private_t **restrict private,
    fang_env_ops_t **restrict ops, fang_reallocator_t realloc)
{
    int res = FANG_OK;

    _fang_env_cpu_t *cpu_private = FANG_CREATE(realloc, _fang_env_cpu_t, 1);
    cpu_private->private.release = _fang_env_cpu_release;

    if(!FANG_ISOK(res = _fang_env_cpu_getinfo(&cpu_private->cpu,
        &cpu_private->ncpu, realloc)))
    {
        goto out;
    }

    /* All physical CPUs are active by default. */
    cpu_private->ncact = cpu_private->ncpu;

    *private = (fang_env_private_t *) cpu_private;
    *ops = &_cpu_ops;

out:
    return res;
}

/* ================ DEFINITIONS END ================ */


/* ================ CPU OPERATORS ================ */

/* Private macro of `_fang_env_cpu_ops_create`. */
#define _DATACPY(ltype, rtype)                                         \
for(size_t i = 0; i < size; i++)                                       \
    ((ltype *) ten->data.dense)[i] = (ltype) ((rtype *) arg->y)[i];

/* Creates and initializes tensor data. */
FANG_HOT FANG_FLATTEN int
    _fang_env_cpu_ops_create(fang_ten_ops_arg_t *restrict arg)
{
    int res = FANG_OK;

    /* The tensor to work with. */
    fang_ten_t *ten = (fang_ten_t *) arg->dest;

    /* There is a very good chance the Environment structure would be valid. */
    fang_env_t *env;
    _fang_env_retrieve(&env, ten->eid);

    /* Calculate size in bytes based on tensor data type. */
    size_t size = (size_t) FANG_G2U(arg->x) * dsiz[(int) ten->dtyp % ndsiz];

    /* Allocate memory to store tensor data. */
    ten->data.dense = (fang_gen) FANG_CREATE(env->realloc, char, size);

    if(FANG_UNLIKELY(ten->data.dense == NULL)) {
        res = -FANG_NOMEM;
        goto out;
    }

    /* Fill out the data. */
    if(FANG_LIKELY(arg->y == NULL)) {
        memset(ten->data.dense, 0, size);
    } else {
        switch(ten->dtyp) {
            case FANG_TEN_DTYPE_INT8: {
                _DATACPY(int8_t, fang_int);
            } break;

            case FANG_TEN_DTYPE_INT16: {
                _DATACPY(int16_t, fang_int);
            } break;

            case FANG_TEN_DTYPE_INT32: {
                _DATACPY(int32_t, fang_int);
            } break;

            case FANG_TEN_DTYPE_INT64: {
                _DATACPY(int64_t, fang_int);
            } break;

            case FANG_TEN_DTYPE_UINT8: {
                _DATACPY(uint8_t, fang_uint);
            } break;

            case FANG_TEN_DTYPE_UINT16: {
                _DATACPY(uint16_t, fang_uint);
            } break;

            case FANG_TEN_DTYPE_UINT32: {
                _DATACPY(uint32_t, fang_uint);
            } break;

            case FANG_TEN_DTYPE_UINT64: {
                _DATACPY(uint64_t, fang_uint);
            } break;

            case FANG_TEN_DTYPE_FLOAT8: {
                for(size_t i = 0; i < size; i++) {
                    ((_fang_float8_t *) ten->data.dense)[i] =
                        _FANG_S2Q((float) ((fang_float *) arg->y)[i]);
                }
            } break;

            case FANG_TEN_DTYPE_FLOAT16: {
                for(size_t i = 0; i < size; i++) {
                    ((_fang_float16_t *) ten->data.dense)[i] =
                        _FANG_S2H((float) ((fang_float *) arg->y)[i]);
                }
            } break;

            case FANG_TEN_DTYPE_BFLOAT16: {
                for(size_t i = 0; i < size; i++) {
                    ((_fang_bfloat16_t *) ten->data.dense)[i] =
                        _FANG_S2BH((float) ((fang_float *) arg->y)[i]);
                }
            } break;

            case FANG_TEN_DTYPE_FLOAT32: {
                _DATACPY(float, fang_float);
            } break;

            case FANG_TEN_DTYPE_FLOAT64: {
                _DATACPY(double, fang_float);
            } break;
        }
    }

out:
    return res;
}

/* ================ CPU OPERATORS END ================ */

