#include <fang/util/buffer.h>
#include <fang/status.h>
#include <fang/tensor.h>
#include <env/cpu/float.h>
#include <platform/env/cpu.h>
#include <string.h>
#include <stdbool.h>

/* ================ FORWARD DECLARATIONS ================ */

/* Creates and initializes dense tensor data. */
FANG_HOT FANG_FLATTEN static int _fang_env_cpu_dense_ops_create(
    fang_ten_ops_arg_t *restrict arg, fang_env_t *env);

/* Prints a dense tensor to a buffer. */
FANG_HOT FANG_FLATTEN static int _fang_env_cpu_dense_ops_print(
    fang_ten_ops_arg_t *restrict arg, fang_env_t *env);

/* Releases a dense tensor. */
FANG_HOT FANG_FLATTEN static int _fang_env_cpu_dense_ops_release(
    fang_ten_ops_arg_t *restrict arg, fang_env_t *env);

/* ================ FORWARD DECLARATIONS END ================ */


/* ================ PRIVATE GLOBALS ================ */

static fang_ten_ops_t _dense = {
    .create = _fang_env_cpu_dense_ops_create,
    .print = _fang_env_cpu_dense_ops_print,
    .rand = _fang_env_cpu_dense_ops_rand,
    .release = _fang_env_cpu_dense_ops_release
};

/* Tensor operators for CPU Environment. */
static fang_env_ops_t _cpu_ops = {
    .dense  = &_dense,
    .sparse = NULL
};

/* Single element data size of each tensor data type. */
/* NOTE: This array conforms to `fang_ten_dtype_t` enum, any changes made to
   that enum should reflect here. */
static const int dsiz[] = { 1, 2, 4, 8, 1, 2, 4, 8, 1, 2, 2, 4, 8 };

/* ================ PRIVATE GLOBALS END ================ */


/* ================ PRIVATE DEFINITIONS ================ */

/* Fully releases a private CPU Environment structures. */
static void _fang_env_cpu_release(void *restrict private,
    fang_reallocator_t realloc)
{
    _fang_env_cpu_t *cpu_env = (_fang_env_cpu_t *) private;

    for(int i = 0; i < cpu_env->ncpu; i++) {
        _fang_cpu_t *cpu = cpu_env->cpu + i;

        FANG_RELEASE(realloc, cpu->task->pool);
        FANG_RELEASE(realloc, cpu->task->arg);
        FANG_RELEASE(realloc, cpu->task);
    }

    FANG_RELEASE(realloc, cpu_env->cpu);
    FANG_RELEASE(realloc, cpu_env);
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

    /* Total number of active processors. */
    cpu_private->ncact = 0;
    for(int i = 0; i < cpu_private->ncpu; i++)
        cpu_private->ncact += cpu_private->cpu[i].nact;

    *private = (fang_env_private_t *) cpu_private;
    *ops = &_cpu_ops;

out:
    return res;
}

/* Changes processor count of a physical CPU. */
int _fang_env_cpu_actproc(fang_env_private_t *private, int pcpu, int nact) {
    int res = FANG_OK;
    _fang_env_cpu_t *cpu_env = (_fang_env_cpu_t *) private;

    if(pcpu < 0) pcpu += cpu_env->ncpu;
    if(pcpu + 1 > cpu_env->ncpu) {
        res = -FANG_INVPCPU;
        goto out;
    }

    _fang_cpu_t *cpu = cpu_env->cpu + pcpu;
    int prev_nact = cpu->nact;
    if(nact < 0 || nact > cpu->nproc)
        res = -FANG_INVPCOUNT;
    else if(nact == 0)
        cpu->nact = cpu->nproc;
    else cpu->nact = nact;

    /* Change global number on active processors. */
    cpu_env->ncact += cpu->nact - prev_nact;

out:
    return res;
}

/* ================ DEFINITIONS END ================ */


/* ================ PRIVATE DEFINITIONS ================ */

/* Temporary buffer size. */
#define _CBUFSIZ           128
/* Render text to temporary character buffer. */
#define _RENDER(f, ...)    snprintf(cbuf, _CBUFSIZ, f, __VA_ARGS__)
/* Flush character buffer. */
#define _FLUSH(v)          fang_buffer_concat(buff, v)

/* Recursive solution for tensor printing. */
FANG_HOT static void _fang_ten_print_recurse(fang_buffer_t *restrict buff,
    fang_ten_dtype_t typ, void *restrict data, uint32_t level, uint32_t
    *restrict sdims, uint16_t ndims, uint32_t *restrict indicies, int padding,
    bool end)
{
    /* Temporary char buffer for immediate print before pushing to buffer. */
    char cbuf[_CBUFSIZ] = { 0 };

    /* At the last dimension; print individual values. */
    if(level == ndims) {
        /* Calculate stride from stridemension. */
        uint32_t stride = 0;
        for(uint16_t i = 0; i < level; i++)
            stride += indicies[i] * (i + 1 < ndims ? sdims[i] : 1);

        fang_buffer_concat(buff, " ");
        switch(typ) {
            case FANG_TEN_DTYPE_INT8:
                _RENDER("%4hhd", ((int8_t *) data)[stride]);
                break;

            case FANG_TEN_DTYPE_INT16:
                _RENDER("%6hd", ((int16_t *) data)[stride]);
                break;

            case FANG_TEN_DTYPE_INT32:
                _RENDER("%11d", ((int32_t *) data)[stride]);
                break;

            case FANG_TEN_DTYPE_INT64:
                _RENDER("%11ld", ((int64_t *) data)[stride]);
                break;

            case FANG_TEN_DTYPE_UINT8:
                _RENDER("%3hhu", ((uint8_t *) data)[stride]);
                break;

            case FANG_TEN_DTYPE_UINT16:
                _RENDER("%5hu", ((uint16_t *) data)[stride]);
                break;

            case FANG_TEN_DTYPE_UINT32:
                _RENDER("%10u", ((uint32_t *) data)[stride]);
                break;

            case FANG_TEN_DTYPE_UINT64:
                _RENDER("%10lu", ((uint64_t *) data)[stride]);
                break;

            case FANG_TEN_DTYPE_FLOAT8: {
                _RENDER("%10.3f",
                    _FANG_Q2S(((_fang_float8_t *) data)[stride]));
            } break;

            case FANG_TEN_DTYPE_FLOAT16: {
                _RENDER("%11.3f",
                    _FANG_H2S(((_fang_float16_t *) data)[stride]));
            } break;

            case FANG_TEN_DTYPE_BFLOAT16: {
                _RENDER("%11.3f",
                    _FANG_BH2S(((_fang_bfloat16_t *) data)[stride]));
            } break;

            case FANG_TEN_DTYPE_FLOAT32:
                _RENDER("%13.3f", ((float *) data)[stride]);
                break;

            case FANG_TEN_DTYPE_FLOAT64:
                _RENDER("%16.4lf", ((double *) data)[stride]);
                break;

            default: break;
        }

        /* Push to buffer. */
        _FLUSH(cbuf);

        return;
    }

    uint32_t dim;
    /* In stridemension the first dimension is stored at the end. */
    if(level == 0)
        dim = sdims[ndims - 1];
    else if(level + 1 == ndims)
        dim = sdims[level - 1];
    else dim = sdims[level - 1] / sdims[level];

    /* Spacing. */
    if(level == 0)
        _RENDER("%*s", padding, "");
    /* Add level count to compensate for extra brace ('[') for each level. */
    else if(indicies[level - 1] != 0)
        _RENDER("%*s", level + padding, "");

    /* Push to buffer. */
    _FLUSH(cbuf);
    _FLUSH("[");

    /* Recursively print rest of the dimensions/levels. */
    for(indicies[level] = 0; indicies[level] < dim; indicies[level]++) {
        _fang_ten_print_recurse(buff, typ, data, level + 1, sdims, ndims,
            indicies, padding, (indicies[level] + 1 == dim));
    }

    /* At the final dimension, printed individual values. Print an extra space
     * to make it look good. */
    if(level + 1 == ndims)
        _FLUSH(" ");
    _FLUSH("]");

    /* Print extra newlines if end of previous dimension. */
    if(FANG_LIKELY(!end && level != 0)) {
        int count = ndims - level, printed = 0;
        while(count--)
            printed += snprintf(cbuf + printed, _CBUFSIZ - printed, "\n");

        _FLUSH(cbuf);
    }
}
#undef _CBUFSIZ
#undef _RENDER
#undef _FLUSH

/* ================ PRIVATE DEFINITIONS END ================ */


/* ================ CPU DENSE OPERATORS ================ */

/* Private macro of `_fang_env_cpu_ops_create`. */
#define _DATACPY(ltype, rtype)                                         \
for(size_t i = 0; i < size; i++)                                       \
    ((ltype *) ten->data.dense)[i] = (ltype) ((rtype *) arg->y)[i];

/* Creates and initializes dense tensor data. */
FANG_HOT FANG_FLATTEN static int _fang_env_cpu_dense_ops_create(
    fang_ten_ops_arg_t *restrict arg, fang_env_t *env)
{
    int res = FANG_OK;

    /* The tensor to work with. */
    fang_ten_t *ten = (fang_ten_t *) arg->dest;

    /* Calculate size in bytes based on tensor data type. */
    size_t size = (ten->sdims != NULL ? (size_t) FANG_G2U(arg->x) :
        1 /* Scalar tensor. */) * dsiz[(int) ten->dtyp];

    /* Allocate memory to store tensor data. */
    ten->data.dense = FANG_CREATE(env->realloc, char, size);

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

            default: break;
        }
    }

out:
    return res;
}

/* Prints a dense tensor to a buffer. */
FANG_HOT FANG_FLATTEN static int _fang_env_cpu_dense_ops_print(
    fang_ten_ops_arg_t *restrict arg, FANG_UNUSED fang_env_t *env)
{
    int res = FANG_OK;

    fang_ten_t *ten = (fang_ten_t *) arg->dest;
    int padding = (int) FANG_G2I(arg->x);
    fang_buffer_t *buff = (fang_buffer_t *) arg->y;

    size_t size = ten->sdims[0] * ten->sdims[ten->ndims - 1];
    /* Progress indicies. */
    uint32_t *indicies = FANG_CREATE(env->realloc, uint32_t, size);
    if(indicies == NULL) {
        res = -FANG_NOMEM;
        goto out;
    }
    memset(indicies, 0, size * sizeof(uint32_t));

    _fang_ten_print_recurse(buff, ten->dtyp, ten->data.dense, 0, ten->sdims,
        ten->ndims, indicies, padding, false);

    /* Push termination character. */
    char term = '\0';
    fang_buffer_add(buff, &term);

    /* Release indicies. */
    FANG_RELEASE(env->realloc, indicies);

out:
    return res;
}

/* Releases a dense tensor. */
FANG_HOT FANG_FLATTEN static int _fang_env_cpu_dense_ops_release(
    fang_ten_ops_arg_t *restrict arg, fang_env_t *env)
{
    /* The tensor to work with. */
    fang_ten_t *ten = (fang_ten_t *) arg->dest;

    /* Release tensor data. */
    FANG_RELEASE(env->realloc, ten->data.dense);

    return FANG_OK;
}

/* ================ CPU DENSE OPERATORS END ================ */

