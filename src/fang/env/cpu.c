#include <fang/status.h>
#include <fang/tensor.h>
#include <platform/env/cpu.h>

/* Include SIMD headers if available. */
#if defined(FANG_USE_AVX512) || defined(FANG_USE_AVX2)
#include <immintrin.h>
#endif  // FANG_USE_AVX512 and FANG_USE_AVX2

/* ================ FORWARD DECLARATIONS ================ */

/* Creates and initializes tensor data. */
FANG_HOT FANG_FLATTEN int
    _fang_env_cpu_ops_create(fang_ten_ops_arg_t *restrict arg);

/* ================ FORWARD DECLARATIONS END ================ */


/* ================ PRIVATE GLOBALS ================ */

static const fang_ten_ops_t _dense = {
    .create = _fang_env_cpu_create
};

/* Tensor operators for CPU Environment. */
static const fang_env_ops_t _cpu_ops = {
    .dense  = _dense,
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

/* Creates and initializes tensor data. */
#define _DATACPY(ltype, rtype)                                   \
for(size_t i = 0; i < size; i++)                                \
    ((ltype *) *dest)[i] = (itype) ((rtype *) arg->data)[i];
int _fang_env_cpu_ops_create(fang_ten_ops_arg_t *restrict arg) {
    int res = FANG_OK;

    /* The tensor to work with. */
    fang_ten_t *ten = (fang_ten_t *) arg->dest;

#if !defined(FANG_USE_AVX512) && !defined(FANG_USE_AVX2)
    /* For CPU Environment, half-precision and quarter-precision is only
       supported when SIMD is enabled. */
    if(ten->dtyp == FANG_TEN_DTYPE_FLOAT8 ||
       ten->dtyp == FANG_TEN_DTYPE_FLOAT16)
    {
        res = -FANG_NOHFLOAT;
        goto out;
    }
#endif  // !FANG_USE_AVX512 and !FANG_USE_AVX2

    /* There is a very good chance the Environment structure would be valid. */
    fang_env_t *env;
    _fang_env_retrieve(&env, ten->eid);

    /* Calculate size in bytes based on tensor data type. */
    size_t size = (size_t) FANG_G2U(arg->x) * dsiz[(int) ten->dtyp % ndsiz];

    /* Allocate memory to store tensor data. */
    ten->data.dense = (fang_gen) FANG_CREATE(env->realloc, size);

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

#ifdef FANG_USE_AVX512
            case FANG_TEN_DTYPE_FLOAT8: {
                _DATACPY(
            } break;

            case FANG_TEN_DTYPE_FLOAT16,
#endif  // FANG_USE_AVX512 or FANG_USE_AVX2
            case FANG_TEN_DTYPE_FLOAT32,
            case FANG_TEN_DTYPE_FLOAT64
        }
    }

out:
    return res;
}

/* ================ CPU OPERATORS END ================ */

