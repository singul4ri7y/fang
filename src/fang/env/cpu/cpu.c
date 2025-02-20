#include <fang/util/buffer.h>
#include <fang/status.h>
#include <fang/tensor.h>
#include <env/cpu/float.h>
#include <env/cpu/gemm.h>
#include <platform/env/cpu.h>
#include <string.h>
#include <stdbool.h>
#include <stdlib.h>
#include <stdalign.h>
#include <omp.h>

/* ================ PRIVATE DATA STRUCTURES ================ */

/* Arguments passed to each kernel . */
typedef struct _fang_cpu_accel_arg {
    /* Parameters. */
    fang_gen_t dest;  // May used to pass resulting tensor
    /* Can be tensor and/or general purpose data. */
    fang_gen_t x;
    fang_gen_t y;
    fang_gen_t z;
    fang_gen_t alpha;
    fang_gen_t beta;
} _fang_cpu_accel_arg_t;

/* Forward declaration of kernel. */
typedef void (*_fang_cpu_accel_t)(_fang_cpu_accel_arg_t *restrict arg);

/* ================ PRIVATE DATA STRUCTURES END ================ */

/* Include dense tensor operation accelerators. */
#include "dense.c.inc"

/* Dummy accelerator for padding. */
static void _dummy_accel(FANG_UNUSED _fang_cpu_accel_arg_t *restrict arg) {}

/* ================ FORWARD DECLARATIONS ================ */

/* CPU operation declarations. */
#define _FANG_ENV_CPU_DENSE_OPS_DECL(operator)                             \
FANG_HOT FANG_FLATTEN static int _fang_env_cpu_dense_ops_##operator(       \
    fang_ten_ops_arg_t *restrict arg);

/* Creates and initializes dense tensor data. */
_FANG_ENV_CPU_DENSE_OPS_DECL(create)

/* Prints a dense tensor to a buffer. */
_FANG_ENV_CPU_DENSE_OPS_DECL(print)

/* Fills a tensor with random numbers. */
_FANG_ENV_CPU_DENSE_OPS_DECL(rand)

/* Adds two tensors. */
_FANG_ENV_CPU_DENSE_OPS_DECL(sum)

/* Subtracts two tensors. */
_FANG_ENV_CPU_DENSE_OPS_DECL(diff)

/* Multiplies two tensors. */
_FANG_ENV_CPU_DENSE_OPS_DECL(mul)

/* Performs GEMM operation between two tensors (this sounds so cool!). */
_FANG_ENV_CPU_DENSE_OPS_DECL(gemm)

/* Scales a tensor. */
_FANG_ENV_CPU_DENSE_OPS_DECL(scale)

/* Fills a tensor with value. */
_FANG_ENV_CPU_DENSE_OPS_DECL(fill)

/* Releases a dense tensor. */
_FANG_ENV_CPU_DENSE_OPS_DECL(release)

/* ================ FORWARD DECLARATIONS END ================ */


/* ================ PRIVATE GLOBALS ================ */

/* Dense tensor operators. */
static fang_ten_ops_t _dense = {
    .create = _fang_env_cpu_dense_ops_create,
    .print = _fang_env_cpu_dense_ops_print,
    .rand = _fang_env_cpu_dense_ops_rand,
    .sum = _fang_env_cpu_dense_ops_sum,
    .diff = _fang_env_cpu_dense_ops_diff,
    .mul = _fang_env_cpu_dense_ops_mul,
    .gemm = _fang_env_cpu_dense_ops_gemm,
    .scale = _fang_env_cpu_dense_ops_scale,
    .fill = _fang_env_cpu_dense_ops_fill,
    .release = _fang_env_cpu_dense_ops_release
};

/* Tensor operators for CPU Environment. */
static fang_env_ops_t _cpu_ops = {
    .dense  = &_dense,
    .sparse = NULL
};

/* Single element data size of each tensor data type. */
/* NOTE: This array conforms to `fang_ten_dtype_t` enum, any changes made to
 *   that enum should reflect here.
 */
static const int dsiz[] = { 1, 2, 4, 8, 1, 2, 4, 8, 1, 2, 2, 4, 8 };

/* NOTE: Array order conforms to `fang_ten_dtype_t` enum. */
_fang_cpu_accel_t _dense_rand[] = {
    _ACCEL_DENSE(rand)
};
_fang_cpu_accel_t _dense_scale[] = {
    _ACCEL_DENSE(scale)
};
_fang_cpu_accel_t _dense_fill[] = {
    _ACCEL_DENSE(fill)
};
_fang_cpu_accel_t _dense_sum[] = {
    _ACCEL_DENSE(sum)
};
_fang_cpu_accel_t _dense_diff[] = {
    _ACCEL_DENSE(diff)
};
_fang_cpu_accel_t _dense_mul[] = {
    _ACCEL_DENSE(mul)
};
_fang_cpu_accel_t _dense_gemm[] = {
    /* Fill dummy accelerators as padding. */
    _dummy_accel, _dummy_accel, _dummy_accel, _dummy_accel,
    _dummy_accel, _dummy_accel, _dummy_accel, _dummy_accel,
    _dummy_accel, _dummy_accel, _dummy_accel,

    // TODO: Add more types
    _fang_dense_accel_gemmf32
};

/* To check difference in tensor randomizer. `_of_ma` = overflow max. Used in
   `_fang_env_cpu_dense_ops_rand`. */
/* NOTE: Array conforms to `fang_ten_dtype_t` enum order. */
fang_uint_t _of_ma[] = { INT8_MAX, INT16_MAX, INT32_MAX, INT64_MAX,
    UINT8_MAX, UINT16_MAX, UINT32_MAX, UINT64_MAX };  // Max
fang_uint_t _of_mi[] = { INT8_MIN, INT16_MIN, INT32_MIN, INT64_MIN };  // Min

/* ================ PRIVATE GLOBALS END ================ */


/* ================ PRIVATE DEFINITIONS ================ */

/* Fully releases a private CPU Environment structures. */
static void _fang_env_cpu_release(void *restrict private,
    fang_reallocator_t realloc)
{
    _fang_env_cpu_t *cpu_env = (_fang_env_cpu_t *) private;
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

    if(!FANG_ISOK(res = _fang_env_cpu_getinfo(&cpu_private->nproc)))
    {
        goto out;
    }

    /* All processors are active at first. */
    cpu_private->nact = cpu_private->nproc;

    *private = (fang_env_private_t *) cpu_private;
    *ops = &_cpu_ops;

    omp_set_num_threads(cpu_private->nact);

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

/* Prints an element to buffer. */
FANG_HOT FANG_INLINE static inline void
    _fang_print_elem_to_buff(fang_buffer_t *buff, int typ, void *data, int idx)
{
    char cbuf[_CBUFSIZ] = { 0 };

    switch(typ) {
        case FANG_TEN_DTYPE_INT8:
            _RENDER("%4hhd", ((int8_t *) data)[idx]);
            break;

        case FANG_TEN_DTYPE_INT16:
            _RENDER("%6hd", ((int16_t *) data)[idx]);
            break;

        case FANG_TEN_DTYPE_INT32:
            _RENDER("%11d", ((int32_t *) data)[idx]);
            break;

        case FANG_TEN_DTYPE_INT64:
            _RENDER("%11ld", ((int64_t *) data)[idx]);
            break;

        case FANG_TEN_DTYPE_UINT8:
            _RENDER("%3hhu", ((uint8_t *) data)[idx]);
            break;

        case FANG_TEN_DTYPE_UINT16:
            _RENDER("%5hu", ((uint16_t *) data)[idx]);
            break;

        case FANG_TEN_DTYPE_UINT32:
            _RENDER("%10u", ((uint32_t *) data)[idx]);
            break;

        case FANG_TEN_DTYPE_UINT64:
            _RENDER("%10lu", ((uint64_t *) data)[idx]);
            break;

        case FANG_TEN_DTYPE_FLOAT8: {
            _RENDER("%10.3f",
                _FANG_Q2S(((_fang_float8_t *) data)[idx]));
        } break;

        case FANG_TEN_DTYPE_FLOAT16: {
            _RENDER("%11.3f",
                _FANG_H2S(((_fang_float16_t *) data)[idx]));
        } break;

        case FANG_TEN_DTYPE_BFLOAT16: {
            _RENDER("%11.3f",
                _FANG_BH2S(((_fang_bfloat16_t *) data)[idx]));
        } break;

        case FANG_TEN_DTYPE_FLOAT32:
            _RENDER("%13.3f", ((float *) data)[idx]);
            break;

        case FANG_TEN_DTYPE_FLOAT64:
            _RENDER("%16.4lf", ((double *) data)[idx]);
            break;

        default: break;
    }

    /* Push to buffer. */
    _FLUSH(cbuf);
}

/* Recursive solution for tensor printing. */
FANG_HOT static void _fang_ten_print_recurse(fang_buffer_t *restrict buff,
    fang_ten_dtype_t typ, void *restrict data, uint32_t level, uint32_t
    *dims, uint32_t *strides, uint16_t ndims, uint32_t *restrict indicies,
    int padding)
{
    /* Temporary char buffer for immediate print before pushing to buffer. */
    char cbuf[_CBUFSIZ] = { 0 };

    /* At the last dimension; print individual values. */
    if(level == ndims) {
        /* Calculate stride. */
        uint32_t stride = 0;
        for(uint16_t i = 0; i < level; i++)
            stride += indicies[i] * strides[i];

        _FLUSH(" ");
        _fang_print_elem_to_buff(buff, (int) typ, data, stride);

        return;
    }

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
    for(indicies[level] = 0; indicies[level] < dims[level]; indicies[level]++) {
        _fang_ten_print_recurse(buff, typ, data, level + 1, dims, strides,
            ndims, indicies, padding);
    }

    /* At the final dimension, printed individual values. Print an extra space
     * to make it look good. */
    if(level + 1 == ndims)
        _FLUSH(" ");
    _FLUSH("]");

    /* Print extra newlines if end of previous dimension. */
    if(FANG_LIKELY(level > 0 && indicies[level - 1] + 1 != dims[level - 1])) {
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
for(size_t i = 0; i < elems; i++)                                      \
    ((ltype *) ten->data.dense)[i] = (ltype) ((rtype *) arg->y)[i];

/* Creates and initializes dense tensor data. */
int _fang_env_cpu_dense_ops_create(fang_ten_ops_arg_t *restrict arg) {
    int res = FANG_OK;

    /* The tensor to work with. */
    fang_ten_t *ten = (fang_ten_t *) arg->dest;
    fang_env_t *env = (fang_env_t *) arg->z;

    /* Calculate size in bytes based on tensor data type. */
    uint32_t elems = (ten->dims != NULL ? (size_t) FANG_G2U(arg->x) :
        1 /* Scalar tensor. */);
    size_t size = elems * dsiz[(int) ten->dtyp];

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
                _DATACPY(int8_t, fang_int_t);
            } break;

            case FANG_TEN_DTYPE_INT16: {
                _DATACPY(int16_t, fang_int_t);
            } break;

            case FANG_TEN_DTYPE_INT32: {
                _DATACPY(int32_t, fang_int_t);
            } break;

            case FANG_TEN_DTYPE_INT64: {
                _DATACPY(int64_t, fang_int_t);
            } break;

            case FANG_TEN_DTYPE_UINT8: {
                _DATACPY(uint8_t, fang_uint_t);
            } break;

            case FANG_TEN_DTYPE_UINT16: {
                _DATACPY(uint16_t, fang_uint_t);
            } break;

            case FANG_TEN_DTYPE_UINT32: {
                _DATACPY(uint32_t, fang_uint_t);
            } break;

            case FANG_TEN_DTYPE_UINT64: {
                _DATACPY(uint64_t, fang_uint_t);
            } break;

            case FANG_TEN_DTYPE_FLOAT8: {
                for(size_t i = 0; i < elems; i++) {
                    ((_fang_float8_t *) ten->data.dense)[i] =
                        _FANG_S2Q((float) ((fang_float_t *) arg->y)[i]);
                }
            } break;

            case FANG_TEN_DTYPE_FLOAT16: {
                for(size_t i = 0; i < elems; i++) {
                    ((_fang_float16_t *) ten->data.dense)[i] =
                        _FANG_S2H((float) ((fang_float_t *) arg->y)[i]);
                }
            } break;

            case FANG_TEN_DTYPE_BFLOAT16: {
                for(size_t i = 0; i < elems; i++) {
                    ((_fang_bfloat16_t *) ten->data.dense)[i] =
                        _FANG_S2BH((float) ((fang_float_t *) arg->y)[i]);
                }
            } break;

            case FANG_TEN_DTYPE_FLOAT32: {
                _DATACPY(float, fang_float_t);
            } break;

            case FANG_TEN_DTYPE_FLOAT64: {
                _DATACPY(double, fang_float_t);
            } break;

            default: break;
        }
    }

out:
    return res;
}

/* Prints a dense tensor to a buffer. */
int _fang_env_cpu_dense_ops_print(fang_ten_ops_arg_t *restrict arg) {
    fang_ten_t *ten = (fang_ten_t *) arg->dest;
    int padding = (int) FANG_G2I(arg->x);
    fang_buffer_t *buff = (fang_buffer_t *) arg->y;

    if(FANG_LIKELY(ten->dims != NULL)) {
        /* Progress indicies. */
        uint32_t indicies[ten->ndims];

        _fang_ten_print_recurse(buff, ten->dtyp, ten->data.dense, 0, ten->dims,
            ten->strides, ten->ndims, indicies, padding);
    } else  // Handle scalar tensor
        _fang_print_elem_to_buff(buff, (int) ten->dtyp, ten->data.dense, 0);

    /* Push termination character. */
    char term = '\0';
    fang_buffer_add(buff, &term);

    return FANG_OK;
}

/* Fills a tensor with random numbers. */
int _fang_env_cpu_dense_ops_rand(fang_ten_ops_arg_t *restrict arg) {
    int res = FANG_OK;

    fang_ten_t *ten = (fang_ten_t *) arg->dest;
    fang_gen_t diff, low = arg->x;

    if(FANG_LIKELY(ten->dtyp >= FANG_TEN_DTYPE_INT8 &&
        ten->dtyp <= FANG_TEN_DTYPE_INT64))
    {
        /* high - low */
        /* Add 1 to make high range inclusive. */
        int64_t intm_diff = FANG_G2I(arg->y) - FANG_G2I(arg->x) + 1;

        /* Check for overflow. */
        if(FANG_UNLIKELY(intm_diff < (int64_t) _of_mi[(int) ten->dtyp] ||
            intm_diff > (int64_t) _of_ma[(int) ten->dtyp]))
        {
            res = -FANG_RANDOF;
            goto out;
        }

        diff = FANG_I2G(intm_diff);
    }
    else if(FANG_LIKELY(ten->dtyp >= FANG_TEN_DTYPE_UINT8 &&
        ten->dtyp <= FANG_TEN_DTYPE_UINT64))
    {
        /* high - low */
        uint64_t intm_diff = FANG_G2U(arg->y) - FANG_G2U(arg->x) + 1;

        /* Check for overflow. */
        if(FANG_UNLIKELY(intm_diff >
            (uint64_t) _of_ma[(int) ten->dtyp]))
        {
            res = -FANG_RANDOF;
            goto out;
        }

        diff = FANG_U2G(intm_diff);
    } else diff = FANG_F2G(FANG_G2F(arg->y) - FANG_G2F(arg->x));

    _fang_cpu_accel_arg_t accel_arg = {
        .dest = (fang_gen_t) ten,
        .x = diff,
        .y = low,
        .z = arg->z  // seed
    };
    _dense_rand[(int) ten->dtyp](&accel_arg);

out:
    return res;
}

/* Scales a tensor. */
int _fang_env_cpu_dense_ops_scale(fang_ten_ops_arg_t *restrict arg) {
    int res = FANG_OK;

    fang_ten_t *ten = (fang_ten_t *) arg->dest;

    _fang_cpu_accel_arg_t accel_arg = {
        .dest = arg->dest,
        .x = arg->x,
    };
    _dense_scale[(int) ten->dtyp](&accel_arg);

    return res;

}

/* Fills a tensor with value. */
int _fang_env_cpu_dense_ops_fill(fang_ten_ops_arg_t *restrict arg) {
    int res = FANG_OK;

    fang_ten_t *ten = (fang_ten_t *) arg->dest;

    _fang_cpu_accel_arg_t accel_arg = {
        .dest = arg->dest,
        .x = arg->x,
    };
    _dense_fill[(int) ten->dtyp](&accel_arg);

    return res;

}

/* Macro to abstract away redundant tensor arithmatic operations. */
#define _FANG_ENV_CPU_DENSE_OPS_ARITH_DEF(operator)                           \
int _fang_env_cpu_dense_ops_##operator(fang_ten_ops_arg_t *restrict arg) {    \
    int res = FANG_OK;                                                        \
                                                                              \
    fang_ten_t *dest = (fang_ten_t *) arg->dest;                              \
    _fang_cpu_accel_arg_t accel_arg = {                                       \
        .dest = arg->dest,                                                    \
        .x = arg->x,                                                          \
        .y = arg->y,                                                          \
        .z = arg->z                                                           \
    };                                                                        \
    _dense_##operator[(int) dest->dtyp](&accel_arg);                          \
                                                                              \
    return res;                                                               \
}

/* Adds two tensor. */
_FANG_ENV_CPU_DENSE_OPS_ARITH_DEF(sum)

/* Subtracts two tensor. */
_FANG_ENV_CPU_DENSE_OPS_ARITH_DEF(diff)

/* Multiplies two tensor. */
_FANG_ENV_CPU_DENSE_OPS_ARITH_DEF(mul)

/* Performs GEMM operation between two tensors (this sounds so cool!). */
int _fang_env_cpu_dense_ops_gemm(fang_ten_ops_arg_t *restrict arg) {
    int res = FANG_OK;

    fang_ten_t *dest = (fang_ten_t *) arg->dest;

    _fang_cpu_accel_arg_t accel_arg = {
        .dest = arg->dest,
        .x = arg->x,
        .y = arg->y,
        .z = arg->z,
        .alpha = arg->alpha,
        .beta = arg->beta
    };
    _dense_gemm[(int) dest->dtyp](&accel_arg);

    return res;
}

/* Releases a dense tensor. */
int _fang_env_cpu_dense_ops_release(fang_ten_ops_arg_t *restrict arg) {
    /* The tensor to work with. */
    fang_ten_t *ten = (fang_ten_t *) arg->dest;
    fang_env_t *env = (fang_env_t *) arg->z;

    /* Release tensor data. */
    FANG_RELEASE(env->realloc, ten->data.dense);

    return FANG_OK;
}

/* ================ CPU DENSE OPERATORS END ================ */

